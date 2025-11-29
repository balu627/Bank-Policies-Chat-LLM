# backend/ingest/build_indexes.py
import os
import pickle
from pathlib import Path
from typing import Dict, List, Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from backend.ingest.config import (
    BASE_DATA_DIR,
    BASE_STORAGE_DIR,
    EMBEDDING_MODEL,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
)
from backend.ingest.pdf_utils import read_pdf_text, chunk_text


def build_corpus_for_folder(folder_name: str) -> Dict[str, Any]:
    """
    Build chunks + metadatas for one corpus (e.g. common, hdfc, sbi).
    """
    data_folder = BASE_DATA_DIR / folder_name
    if not data_folder.exists():
        print(f"[WARN] Data folder does not exist: {data_folder}")
        return {"chunks": [], "metadatas": []}

    all_chunks: List[str] = []
    all_metadatas: List[Dict[str, Any]] = []

    for filename in os.listdir(data_folder):
        if not filename.lower().endswith(".pdf"):
            continue

        pdf_path = data_folder / filename
        print(f"[{folder_name}] Reading: {pdf_path}")
        text = read_pdf_text(pdf_path)

        chunks = chunk_text(
            text=text,
            chunk_size=DEFAULT_CHUNK_SIZE,
            overlap=DEFAULT_CHUNK_OVERLAP,
        )

        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metadatas.append(
                {
                    "source_file": filename,
                    "chunk_id": i,
                    "folder": folder_name,
                }
            )

    print(f"[{folder_name}] Total chunks: {len(all_chunks)}")
    return {"chunks": all_chunks, "metadatas": all_metadatas}


def build_index(chunks: List[str], model_name: str) -> faiss.IndexFlatL2:
    """Create embeddings and FAISS index."""
    if not chunks:
        raise ValueError("No chunks to index.")

    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    print("Encoding chunks...")
    embeddings = model.encode(chunks, show_progress_bar=True)
    embeddings = np.asarray(embeddings).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    return index


def save_index_and_meta(
    folder_name: str,
    index: faiss.IndexFlatL2,
    chunks: List[str],
    metadatas: List[Dict[str, Any]],
):
    """
    Save index and metadata into storage/<folder_name>/
    """
    storage_folder = BASE_STORAGE_DIR / folder_name
    storage_folder.mkdir(parents=True, exist_ok=True)

    index_path = storage_folder / "index.faiss"
    meta_path = storage_folder / "metadata.pkl"

    print(f"[{folder_name}] Saving FAISS index -> {index_path}")
    faiss.write_index(index, str(index_path))

    print(f"[{folder_name}] Saving metadata -> {meta_path}")
    with open(meta_path, "wb") as f:
        pickle.dump(
            {
                "chunks": chunks,
                "metadatas": metadatas,
            },
            f,
        )


def main():
    # Detect corpus folders: all subfolders under data/
    if not BASE_DATA_DIR.exists():
        raise FileNotFoundError(f"DATA_DIR not found: {BASE_DATA_DIR}")

    folder_names = [
        name
        for name in os.listdir(BASE_DATA_DIR)
        if (BASE_DATA_DIR / name).is_dir()
    ]

    if not folder_names:
        print("No folders found under data/. Please create common/hdfc/... etc.")
        return

    print(f"Found corpora: {folder_names}")
    for folder in folder_names:
        print(f"\n=== Building corpus for folder: {folder} ===")
        corpus = build_corpus_for_folder(folder)
        chunks = corpus["chunks"]
        metadatas = corpus["metadatas"]

        if not chunks:
            print(f"[{folder}] No chunks found, skipping index build.")
            continue

        index = build_index(chunks, EMBEDDING_MODEL)
        save_index_and_meta(folder, index, chunks, metadatas)

    print("\nAll indexes built.")


if __name__ == "__main__":
    main()
