"""build_index.py — Build and persist Dense + TF-IDF search indexes for EduDocAI.

Usage:
    python scripts/build_index.py [--chunk-size 400] [--chunk-overlap 60]

What it does:
1. Loads all PDFs from data/sample_documents/.
2. Chunks documents with configurable size and overlap.
3. Builds a FAISS dense index (sentence-transformers/all-MiniLM-L6-v2).
4. Builds a TF-IDF sparse index.
5. Saves index artifacts to artifacts/indexes/.
6. Prints build statistics.
"""

import sys
import time
import argparse
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import faiss
import numpy as np
from config.settings import settings, SAMPLE_DOCS_DIR, BASE_DIR
from src.ingestion.document_loader import DocumentLoader
from src.preprocessing.chunking import TextChunker
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.tfidf_retriever import TfidfRetriever
from src.utils.logger import logger

# Artifacts directory
ARTIFACTS_DIR = BASE_DIR / "artifacts" / "indexes"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def build_indexes(chunk_size: int = 400, chunk_overlap: int = 60) -> None:
    """Build and save Dense + TF-IDF indexes."""
    print(f"\n{'='*60}")
    print("  EduDocAI — Build Search Index")
    print(f"{'='*60}")
    print(f"Chunk size    : {chunk_size} words")
    print(f"Chunk overlap : {chunk_overlap} words")
    print(f"Artifacts dir : {ARTIFACTS_DIR}")

    # 1. Load documents
    print("\n[1/4] Loading documents from data/sample_documents/ ...")
    loader = DocumentLoader()
    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    pdf_files = sorted(SAMPLE_DOCS_DIR.glob("*.pdf"))
    if not pdf_files:
        pdf_files = sorted(SAMPLE_DOCS_DIR.glob("*.txt"))
    if not pdf_files:
        print("[ERROR] No documents found. Add PDFs to data/sample_documents/")
        sys.exit(1)

    all_pages = []
    for f in pdf_files:
        try:
            pages = loader.load_from_path(f)
            all_pages.extend(pages)
            print(f"  Loaded: {f.name} ({len(pages)} pages)")
        except Exception as e:
            print(f"  [WARN] Skipping {f.name}: {e}")

    # 2. Chunk
    print(f"\n[2/4] Chunking {len(all_pages)} pages ...")
    t0 = time.perf_counter()
    chunks = chunker.chunk_documents(all_pages)
    chunk_time = time.perf_counter() - t0
    print(f"  Created {len(chunks)} chunks in {chunk_time:.3f}s")

    # 3. Build Dense Index
    print("\n[3/4] Building Dense FAISS index ...")
    t0 = time.perf_counter()
    dense = DenseRetriever()
    dense.build_index(chunks)
    dense_time = time.perf_counter() - t0
    print(f"  FAISS index built in {dense_time:.3f}s")

    # Save dense index artifacts
    faiss_path = ARTIFACTS_DIR / "faiss_index.bin"
    chunks_path = ARTIFACTS_DIR / "chunks.pkl"
    emb_path    = ARTIFACTS_DIR / "embeddings.npy"

    faiss.write_index(dense.index, str(faiss_path))
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)
    np.save(str(emb_path), dense.embeddings)
    print(f"  Saved FAISS index  -> {faiss_path}")
    print(f"  Saved chunks       -> {chunks_path}")
    print(f"  Saved embeddings   -> {emb_path}")

    # 4. Build TF-IDF Index
    print("\n[4/4] Building TF-IDF index ...")
    t0 = time.perf_counter()
    tfidf = TfidfRetriever()
    tfidf.build_index(chunks)
    tfidf_time = time.perf_counter() - t0
    print(f"  TF-IDF index built in {tfidf_time:.3f}s")

    tfidf_path = ARTIFACTS_DIR / "tfidf_retriever.pkl"
    with open(tfidf_path, "wb") as f:
        pickle.dump(tfidf, f)
    print(f"  Saved TF-IDF index -> {tfidf_path}")

    # Summary
    print(f"\n{'='*60}")
    print("  Build Complete")
    print(f"{'='*60}")
    print(f"Documents  : {len(pdf_files)}")
    print(f"Pages      : {len(all_pages)}")
    print(f"Chunks     : {len(chunks)}")
    word_counts = [c.word_count for c in chunks]
    print(f"Avg words/chunk : {sum(word_counts)/len(word_counts):.1f}")
    print(f"Dense build time: {dense_time:.3f}s")
    print(f"TF-IDF build time: {tfidf_time:.3f}s")
    print("\nRun `python scripts/run_evaluation.py` to evaluate.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build EduDocAI search indexes.")
    parser.add_argument("--chunk-size",    type=int, default=400,
                        help="Words per chunk (default: 400)")
    parser.add_argument("--chunk-overlap", type=int, default=60,
                        help="Overlap words between chunks (default: 60)")
    args = parser.parse_args()
    build_indexes(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)


if __name__ == "__main__":
    main()
