"""prepare_dataset.py — Download or verify the evaluation dataset for EduDocAI.

Usage:
    python scripts/prepare_dataset.py

What it does:
1. Verifies that the evaluation dataset (data/evaluation/eval_dataset.json) exists.
2. Checks that sample documents are present in data/sample_documents/.
3. Prints a dataset manifest with page counts and word statistics.
4. Performs duplicate detection across chunks.
5. Analyses empty/missing pages.
"""

import sys
import json
import hashlib
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from config.settings import settings, DATA_DIR, SAMPLE_DOCS_DIR, EVAL_DIR
from src.ingestion.document_loader import DocumentLoader
from src.preprocessing.chunking import TextChunker
from src.utils.logger import logger


def print_section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print("=" * 60)


def load_eval_dataset() -> list:
    """Load and validate the evaluation dataset."""
    eval_path = EVAL_DIR / "eval_dataset.json"
    if not eval_path.exists():
        print(f"[ERROR] Evaluation dataset not found at: {eval_path}")
        print("Please ensure data/evaluation/eval_dataset.json is present.")
        return []

    with open(eval_path, encoding="utf-8") as f:
        dataset = json.load(f)

    print(f"[OK] Evaluation dataset: {len(dataset)} questions loaded.")
    categories = {}
    for item in dataset:
        cat = item.get("category", "Unknown")
        categories[cat] = categories.get(cat, 0) + 1
    print("\nQuestion Category Breakdown:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat:<40} {count} questions")
    return dataset


def analyse_documents() -> None:
    """Load sample documents and print manifest / quality analysis."""
    pdf_files = sorted(SAMPLE_DOCS_DIR.glob("*.pdf"))
    txt_files = sorted(SAMPLE_DOCS_DIR.glob("*.txt"))
    all_files = pdf_files + txt_files

    if not all_files:
        print("[WARN] No documents found in data/sample_documents/")
        print("  Please add PDF files to data/sample_documents/ before running evaluation.")
        return

    loader = DocumentLoader()
    chunker = TextChunker()

    manifest_rows = []
    all_chunks = []
    all_pages = []

    for f in all_files:
        try:
            pages = loader.load_from_path(f)
            chunks = chunker.chunk_documents(pages)
            all_pages.extend(pages)
            all_chunks.extend(chunks)

            total_chars = sum(len(p.text) for p in pages)
            empty_pages = sum(1 for p in pages if len(p.text.strip()) < 10)
            total_words = sum(c.word_count for c in chunks)

            manifest_rows.append({
                "Filename": f.name,
                "Extension": f.suffix,
                "Size (KB)": round(f.stat().st_size / 1024, 1),
                "Pages": len(pages),
                "Empty Pages": empty_pages,
                "Chunks": len(chunks),
                "Total Words": total_words,
                "Total Chars": total_chars,
                "Avg Words/Chunk": round(total_words / max(len(chunks), 1), 1),
            })
            print(f"  [OK] {f.name}: {len(pages)} pages, {len(chunks)} chunks, {total_words:,} words")
        except Exception as e:
            print(f"  [ERROR] {f.name}: {e}")
            manifest_rows.append({
                "Filename": f.name, "Extension": f.suffix,
                "Size (KB)": round(f.stat().st_size / 1024, 1),
                "Pages": 0, "Empty Pages": 0, "Chunks": 0,
                "Total Words": 0, "Total Chars": 0, "Avg Words/Chunk": 0,
            })

    # Print manifest table
    df = pd.DataFrame(manifest_rows)
    print("\nDataset Manifest:")
    print(df.to_string(index=False))

    # Duplicate chunk detection
    print_section("Duplicate Chunk Detection")
    chunk_hashes = {}
    duplicates = 0
    for chunk in all_chunks:
        h = hashlib.md5(chunk.chunk_text.encode("utf-8")).hexdigest()
        if h in chunk_hashes:
            duplicates += 1
        else:
            chunk_hashes[h] = chunk.chunk_id
    print(f"Total chunks: {len(all_chunks)}")
    print(f"Duplicate chunks: {duplicates} ({duplicates/max(len(all_chunks),1)*100:.1f}%)")
    print(f"Unique chunks: {len(chunk_hashes)}")

    # Empty page analysis
    print_section("Empty / Short Page Analysis")
    empty = [p for p in all_pages if len(p.text.strip()) < 10]
    short = [p for p in all_pages if 10 <= len(p.text.strip()) < 100]
    print(f"Total pages: {len(all_pages)}")
    print(f"Empty pages (< 10 chars): {len(empty)}")
    print(f"Short pages (10-100 chars): {len(short)}")
    for p in empty:
        print(f"  [EMPTY] {p.source} page {p.page}")

    # Word statistics
    print_section("Text Statistics")
    if all_chunks:
        word_counts = [c.word_count for c in all_chunks]
        print(f"Avg words/chunk : {sum(word_counts)/len(word_counts):.1f}")
        print(f"Min words/chunk : {min(word_counts)}")
        print(f"Max words/chunk : {max(word_counts)}")
        print(f"Total words     : {sum(word_counts):,}")


def main() -> None:
    print_section("EduDocAI — Dataset Preparation & Audit")
    print(f"Project root : {DATA_DIR.parent}")
    print(f"Data dir     : {DATA_DIR}")
    print(f"Sample docs  : {SAMPLE_DOCS_DIR}")
    print(f"Eval dir     : {EVAL_DIR}")

    print_section("Evaluation Dataset")
    load_eval_dataset()

    print_section("Document Analysis")
    analyse_documents()

    print_section("Dataset Preparation Complete")
    print("Run `python scripts/build_index.py` to build search indexes.")
    print("Run `python scripts/run_evaluation.py` to execute evaluation.")


if __name__ == "__main__":
    main()
