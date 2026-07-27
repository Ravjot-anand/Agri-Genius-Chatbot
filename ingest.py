"""
ingest.py -- Agri-Genius Knowledge Ingestion Pipeline

Reads agricultural data from MULTIPLE sources (PDFs + text files) in the
data/ folder, chunks them, generates embeddings with sentence-transformers,
and persists everything into a local ChromaDB collection.

Supported formats: .pdf, .txt
"""

import os
import glob
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Memory Optimization for Low-RAM Environments (Render 512MB) ─────────────
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import torch
torch.set_num_threads(1)

from sentence_transformers import SentenceTransformer

# ── Configuration ──────────────────────────────────────────────────────────
DATA_DIR = "data"
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "agri_knowledge"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


# ── Document Loaders ──────────────────────────────────────────────────────
def load_text_file(path: str) -> str:
    """Read a plain text file."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_pdf_file(path: str) -> str:
    """Extract text from a PDF using PyMuPDF (fitz)."""
    import fitz  # PyMuPDF

    text_parts = []
    with fitz.open(path) as doc:
        for page_num, page in enumerate(doc, 1):
            page_text = page.get_text("text")
            if page_text.strip():
                text_parts.append(f"[Page {page_num}]\n{page_text}")
    return "\n\n".join(text_parts)


def load_document(path: str) -> tuple[str, str]:
    """
    Load a document based on its file extension.
    Returns (text_content, file_type).
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return load_pdf_file(path), "pdf"
    elif ext == ".txt":
        return load_text_file(path), "txt"
    else:
        print(f"    [SKIP] Unsupported format: {path}")
        return "", "unknown"


def discover_documents(data_dir: str) -> list[str]:
    """Find all .txt and .pdf files in the data directory (recursive)."""
    patterns = ["**/*.txt", "**/*.pdf"]
    files = []
    for pattern in patterns:
        files.extend(glob.glob(os.path.join(data_dir, pattern), recursive=True))
    return sorted(set(files))


def chunk_text(text: str) -> list[str]:
    """Split text into overlapping chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)


def main():
    # 1. Discover documents
    print(f"[*] Scanning '{DATA_DIR}/' for .txt and .pdf files...")
    doc_paths = discover_documents(DATA_DIR)

    if not doc_paths:
        print(f"[ERROR] No .txt or .pdf files found in '{DATA_DIR}/'.")
        print("        Place your agricultural PDFs/text files there and re-run.")
        return

    print(f"    Found {len(doc_paths)} document(s):")
    for p in doc_paths:
        print(f"      - {p}")

    # 2. Load and chunk all documents
    all_chunks = []
    all_metadatas = []

    for doc_path in doc_paths:
        print(f"\n[*] Processing: {doc_path}")
        text, file_type = load_document(doc_path)

        if not text.strip():
            print(f"    [SKIP] Empty or unreadable: {doc_path}")
            continue

        print(f"    Loaded {len(text):,} characters ({file_type})")
        chunks = chunk_text(text)
        print(f"    Chunked into {len(chunks)} pieces")

        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metadatas.append({
                "source": os.path.basename(doc_path),
                "source_path": doc_path,
                "file_type": file_type,
                "chunk_index": i,
            })

    if not all_chunks:
        print("[ERROR] No text could be extracted from any document.")
        return

    print(f"\n[*] Total: {len(all_chunks)} chunks from {len(doc_paths)} document(s)")

    # 3. Initialize embedding model
    print(f"[*] Loading embedding model: {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # 4. Generate embeddings
    print("[*] Generating embeddings...")
    embeddings = model.encode(all_chunks, show_progress_bar=True).tolist()

    # 5. Persist to ChromaDB
    print(f"[*] Persisting to ChromaDB at {CHROMA_DIR}...")
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Delete existing collection if it exists (for re-runs)
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"    Deleted existing collection '{COLLECTION_NAME}'.")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # Add documents with pre-computed embeddings
    collection.add(
        ids=[f"chunk_{i}" for i in range(len(all_chunks))],
        documents=all_chunks,
        embeddings=embeddings,
        metadatas=all_metadatas,
    )

    print(f"\n[OK] Successfully ingested {len(all_chunks)} chunks into '{COLLECTION_NAME}'.")
    print(f"     Sources: {', '.join(set(m['source'] for m in all_metadatas))}")
    print(f"     ChromaDB persisted at: {os.path.abspath(CHROMA_DIR)}")


if __name__ == "__main__":
    main()
