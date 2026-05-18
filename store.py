import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from config import DATA_DIR

EMBED_MODEL = SentenceTransformer('all-MiniLM-L6-v2')


def find_existing_pdf(path: str):
    """Return the pdf_id if this path was already ingested, else None."""
    if not os.path.exists(DATA_DIR):
        return None
    for pdf_id in os.listdir(DATA_DIR):
        content_path = os.path.join(DATA_DIR, pdf_id, "content.json")
        if os.path.isfile(content_path):
            with open(content_path) as f:
                data = json.load(f)
            if data["original_path"] == path:
                return pdf_id
    return None


def save_chunks_to_faiss(pdf_id: str, pages_data: list):
    """Chunk the text, embed it, and persist a FAISS index."""
    all_text = " ".join(p["text"] for p in pages_data)

    chunk_size = 500
    chunks = [all_text[i:i + chunk_size] for i in range(0, len(all_text), chunk_size)]

    vectors = EMBED_MODEL.encode(chunks)

    index = faiss.IndexFlatL2(384)
    index.add(vectors)

    faiss.write_index(index, os.path.join(DATA_DIR, pdf_id, "index.faiss"))

    with open(os.path.join(DATA_DIR, pdf_id, "chunks.json"), "w") as f:
        json.dump(chunks, f)

    return len(chunks)


def search_faiss(pdf_id: str, question: str, top_k: int = 3):
    """Return the top-k chunks most similar to the question."""
    index_path = os.path.join(DATA_DIR, pdf_id, "index.faiss")
    chunks_path = os.path.join(DATA_DIR, pdf_id, "chunks.json")

    if not os.path.exists(index_path):
        return []

    index = faiss.read_index(index_path)
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    q_vector = EMBED_MODEL.encode([question])
    _, indices = index.search(np.array(q_vector), top_k)

    return [chunks[idx] for idx in indices[0] if idx != -1]