# backend/ingest/pdf_utils.py
from pathlib import Path
from typing import List
from pypdf import PdfReader


def read_pdf_text(pdf_path: Path) -> str:
    """Extract raw text from a PDF file."""
    reader = PdfReader(str(pdf_path))
    texts = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        texts.append(page_text)
    return "\n".join(texts)


def chunk_text(
    text: str,
    chunk_size: int,
    overlap: int,
) -> List[str]:
    """Split text into overlapping character chunks."""
    chunks: List[str] = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start += chunk_size - overlap

    return chunks
