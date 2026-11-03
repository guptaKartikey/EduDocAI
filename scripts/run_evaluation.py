"""run_evaluation.py — Execute the full EduDocAI evaluation pipeline and print results.

Usage:
    python scripts/run_evaluation.py [--top-k 3] [--use-ollama]

What it does:
1. Loads the evaluation dataset (data/evaluation/eval_dataset.json).
2. Indexes all sample documents.
3. Runs Dense and TF-IDF retrieval on every test question.
4. Computes Recall@K, MRR, Token F1, Citation Accuracy, Abstention Accuracy.
5. Saves results to results/metrics/.
6. Prints a formatted summary.
"""

import sys
import json
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings, METRICS_DIR
from src.evaluation.evaluator import RAGEvaluator
from src.utils.logger import logger


def print_section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print("=" * 60)


def print_metric_block(label: str, metrics: dict) -> None:
    print(f"\n  {label}")
    print(f"  {'-'*40}")
    print(f"  Recall@1              : {metrics['mean_recall@1']:.4f}")
    print(f"  Recall@3              : {metrics['mean_recall@3']:.4f}")
    print(f"  Recall@5              : {metrics['mean_recall@5']:.4f}")
    print(f"  MRR                   : {metrics['mrr']:.4f}")
    print(f"  Token F1              : {metrics['token_f1']:.4f}")
    print(f"  Citation Accuracy     : {metrics['citation_accuracy']:.4f}")
    print(f"  Abstention Accuracy   : {metrics['abstention_accuracy']:.4f}")
    print(f"  Avg Groundedness      : {metrics['avg_groundedness']:.4f}")
    print(f"  Avg Retrieval Latency : {metrics['avg_retrieval_latency_ms']:.2f} ms")
    print(f"  Avg Total Latency     : {metrics['avg_total_latency_ms']:.2f} ms")


def run(top_k: int = 3, use_ollama: bool = False) -> None:
    print_section("EduDocAI — Evaluation Pipeline")
    print(f"Top-K           : {top_k}")
    print(f"Use Ollama LLM  : {use_ollama}")

    evaluator = RAGEvaluator()

    print("\n[1/3] Loading evaluation dataset ...")
    t0 = time.perf_counter()
    dataset = evaluator.load_dataset()
    print(f"  Loaded {len(dataset)} test questions in {time.perf_counter()-t0:.3f}s")

    print("\n[2/3] Indexing sample documents ...")
    t0 = time.perf_counter()
    n_chunks = evaluator.index_sample_documents()
    print(f"  Indexed {n_chunks} chunks in {time.perf_counter()-t0:.3f}s")

    print("\n[3/3] Running evaluation ...")
    t0 = time.perf_counter()
    summary = evaluator.run_evaluation(top_k=top_k, use_ollama=use_ollama)
    eval_time = time.perf_counter() - t0
    print(f"  Evaluation completed in {eval_time:.2f}s")

    # Print results
    print_section("Evaluation Results")
    print(f"  Dataset size: {summary['dataset_size']} questions")

    print_metric_block("TF-IDF Baseline", summary["tfidf_retrieval"])
    print_metric_block("Dense Embedding RAG", summary["dense_retrieval"])

    # Category breakdown
    print_section("Category Breakdown (Dense RAG)")
    cat = summary.get("category_breakdown", {})
    for category, vals in cat.items():
        print(f"  {category:<42} Recall@3={vals.get('recall@3',0):.2f}  "
              f"F1={vals.get('f1',0):.2f}  Abstention={vals.get('abstention_accuracy',0):.2f}")

    # Improvement summary
    print_section("Dense RAG vs TF-IDF — Delta")
    d = summary["dense_retrieval"]
    t = summary["tfidf_retrieval"]
    metrics_to_compare = ["mean_recall@1", "mean_recall@3", "mrr", "token_f1",
                           "citation_accuracy", "abstention_accuracy"]
    for m in metrics_to_compare:
        delta = d[m] - t[m]
        sign  = "+" if delta >= 0 else ""
        print(f"  {m:<30} {sign}{delta:.4f}")

    # File locations
    print_section("Output Files")
    print(f"  Summary JSON     : {METRICS_DIR / 'evaluation_summary.json'}")
    print(f"  Dense CSV        : {METRICS_DIR / 'dense_evaluation_details.csv'}")
    print(f"  TF-IDF CSV       : {METRICS_DIR / 'tfidf_evaluation_details.csv'}")
    print("\nOpen the Streamlit app -> Evaluation page to view the dashboard.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EduDocAI evaluation pipeline.")
    parser.add_argument("--top-k",      type=int,  default=3,
                        help="Number of chunks used for answer generation (default: 3)")
    parser.add_argument("--use-ollama", action="store_true",
                        help="Enable local Ollama LLM for generation (default: extractive mode)")
    args = parser.parse_args()
    run(top_k=args.top_k, use_ollama=args.use_ollama)


if __name__ == "__main__":
    main()
