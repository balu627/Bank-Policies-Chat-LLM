# backend/api/retrieval.py
import os
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from config.settings import (
    STORAGE_DIR,
    EMBEDDING_MODEL_NAME,
    TOP_K_PER_INDEX,
    MAX_BANKS_WHEN_NO_BANK_SPECIFIED,
)


class CorpusIndex:
    def __init__(self, name: str, index: faiss.IndexFlatL2, chunks, metadatas):
        self.name = name           # e.g. "common", "hdfc"
        self.index = index
        self.chunks = chunks
        self.metadatas = metadatas


class RetrievalEngine:
    def __init__(self):
        self._corpora: Dict[str, CorpusIndex] = {}
        self._embedding_model: Optional[SentenceTransformer] = None
        self._load_all_corpora()

    def _load_embedding_model(self):
        if self._embedding_model is None:
            self._embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    def _load_all_corpora(self):
        if not STORAGE_DIR.exists():
            raise FileNotFoundError(f"STORAGE_DIR not found: {STORAGE_DIR}")

        for name in os.listdir(STORAGE_DIR):
            folder_path = STORAGE_DIR / name
            if not folder_path.is_dir():
                continue

            index_path = folder_path / "index.faiss"
            meta_path = folder_path / "metadata.pkl"

            if not index_path.exists() or not meta_path.exists():
                print(f"[WARN] Skipping corpus {name}, missing index/meta.")
                continue

            print(f"[Retrieval] Loading corpus '{name}'")
            index = faiss.read_index(str(index_path))
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)

            self._corpora[name] = CorpusIndex(
                name=name,
                index=index,
                chunks=meta["chunks"],
                metadatas=meta["metadatas"],
            )

        print(f"[Retrieval] Loaded corpora: {list(self._corpora.keys())}")

    @property
    def available_banks(self) -> List[str]:
        # All corpora except "common" are banks
        return [name for name in self._corpora.keys() if name.lower() != "common"]

    def _encode_query(self, text: str) -> np.ndarray:
        self._load_embedding_model()
        emb = self._embedding_model.encode([text])
        return np.asarray(emb).astype("float32")

    def _search_single_corpus(
        self,
        corpus: CorpusIndex,
        query_embedding: np.ndarray,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        distances, indices = corpus.index.search(query_embedding, top_k)
        distances = distances[0]
        indices = indices[0]

        results: List[Dict[str, Any]] = []
        for dist, idx in zip(distances, indices):
            if idx == -1:
                continue
            meta = corpus.metadatas[idx]
            chunk_text = corpus.chunks[idx]

            # Merge with neighbors (to repair cut text)
            merged_text = self._merge_neighbors(corpus, idx)

            score = float(1 / (1 + dist))  # smaller distance -> higher score
            results.append(
                {
                    "score": score,
                    "bank": corpus.name,
                    "document_name": meta.get("source_file", "unknown"),
                    "chunk_id": meta.get("chunk_id", int(idx)),
                    "raw_chunk": chunk_text,
                    "merged_text": merged_text,
                }
            )
        return results

    def _merge_neighbors(self, corpus: CorpusIndex, idx: int) -> str:
        """
        Merge current chunk with previous and next if same document.
        This helps repair cut-off sentences.
        """
        current_meta = corpus.metadatas[idx]
        current_file = current_meta.get("source_file", None)
        current_chunk_id = current_meta.get("chunk_id", None)

        texts = []

        # Previous
        if idx - 1 >= 0:
            prev_meta = corpus.metadatas[idx - 1]
            if (
                prev_meta.get("source_file") == current_file
                and prev_meta.get("chunk_id") == current_chunk_id - 1
            ):
                texts.append(corpus.chunks[idx - 1])

        # Current
        texts.append(corpus.chunks[idx])

        # Next
        if idx + 1 < len(corpus.chunks):
            next_meta = corpus.metadatas[idx + 1]
            if (
                next_meta.get("source_file") == current_file
                and next_meta.get("chunk_id") == current_chunk_id + 1
            ):
                texts.append(corpus.chunks[idx + 1])

        merged = "\n".join(texts)
        return merged

    def retrieve(
        self,
        question: str,
        bank: Optional[str],
        top_k_per_index: int | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Returns a list of dicts:
        {
          "bank": "...",
          "document_name": "...",
          "merged_text": "...",
          "score": float
        }
        """
        if top_k_per_index is None:
            top_k_per_index = TOP_K_PER_INDEX

        query_emb = self._encode_query(question)
        corpora_to_search: List[CorpusIndex] = []

        common_corpus = self._corpora.get("common")
        if bank:
            # Specific bank + common
            bank_key = bank.lower()
            # corpora keys are folder names, so assume they are stored lowercase
            bank_corpus = None
            for name, c in self._corpora.items():
                if name.lower() == bank_key:
                    bank_corpus = c
                    break

            if bank_corpus:
                corpora_to_search.append(bank_corpus)
            if common_corpus:
                corpora_to_search.append(common_corpus)
        else:
            # No bank specified: search common + all banks
            if common_corpus:
                corpora_to_search.append(common_corpus)
            for name, c in self._corpora.items():
                if name.lower() == "common":
                    continue
                corpora_to_search.append(c)

        all_results: List[Dict[str, Any]] = []
        for corpus in corpora_to_search:
            corpus_results = self._search_single_corpus(
                corpus, query_emb, top_k_per_index
            )
            all_results.extend(corpus_results)

        # Sort by score DESC
        all_results.sort(key=lambda r: r["score"], reverse=True)

        if bank:
            # We only need top N overall if specific bank
            return all_results[: top_k_per_index * len(corpora_to_search)]

        # No bank: we want:
        # - common docs, plus
        # - up to MAX_BANKS_WHEN_NO_BANK_SPECIFIED banks
        # We'll keep best common chunks, and best from top few banks.
        common_results = [r for r in all_results if r["bank"].lower() == "common"]
        other_results = [r for r in all_results if r["bank"].lower() != "common"]

        # Group other_results by bank
        bank_to_results: Dict[str, List[Dict[str, Any]]] = {}
        for r in other_results:
            bank_to_results.setdefault(r["bank"], []).append(r)

        # Sort banks by best score
        bank_scores: List[Tuple[str, float]] = []
        for bank_name, rs in bank_to_results.items():
            best_score = max(r["score"] for r in rs)
            bank_scores.append((bank_name, best_score))
        bank_scores.sort(key=lambda x: x[1], reverse=True)

        selected_banks = [
            name for name, _ in bank_scores[:MAX_BANKS_WHEN_NO_BANK_SPECIFIED]
        ]

        filtered_results: List[Dict[str, Any]] = []
        filtered_results.extend(common_results[: top_k_per_index])

        for b in selected_banks:
            filtered_results.extend(bank_to_results[b][:top_k_per_index])

        # Sort again by score
        filtered_results.sort(key=lambda r: r["score"], reverse=True)
        return filtered_results


retrieval_engine = RetrievalEngine()
