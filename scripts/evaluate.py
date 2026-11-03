"""CLI script to run quantitative evaluation and generate visual EDA reports."""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
from config.settings import settings, METRICS_DIR
from src.evaluation.evaluator import RAGEvaluator
from scripts.generate_eda_charts import generate_all_charts
from src.utils.logger import logger


def main():
    parser = argparse.ArgumentParser(description="EduDocAI Benchmark Evaluator")
    parser.add_argument("--top_k", type=int, default=3, help="Top-K chunks for retrieval")
    parser.add_argument("--use_ollama", action="store_true", help="Enable Ollama LLM for answer generation")
    args = parser.parse_args()

    print("==================================================")
    print("           EduDocAI Benchmark Evaluator           ")
    print("==================================================")
    print(f"Configuration: Top-K={args.top_k}, Use Ollama={args.use_ollama}")
    print("Indexing documents and executing test suite...")

    evaluator = RAGEvaluator()
    summary = evaluator.run_evaluation(top_k=args.top_k, use_ollama=args.use_ollama)

    print("\n--- BENCHMARK RESULTS SUMMARY ---")
    dense = summary["dense_retrieval"]
    tfidf = summary["tfidf_retrieval"]

    print(f"Total Test Questions: {summary['dataset_size']}\n")
    print(f"{'Metric':<25} | {'Dense (all-MiniLM-L6-v2)':<25} | {'Sparse (TF-IDF)':<20}")
    print("-" * 75)
    print(f"{'Recall@1':<25} | {dense['mean_recall@1']:<25.4f} | {tfidf['mean_recall@1']:<20.4f}")
    print(f"{'Recall@3':<25} | {dense['mean_recall@3']:<25.4f} | {tfidf['mean_recall@3']:<20.4f}")
    print(f"{'Recall@5':<25} | {dense['mean_recall@5']:<25.4f} | {tfidf['mean_recall@5']:<20.4f}")
    print(f"{'Mean Reciprocal Rank':<25} | {dense['mrr']:<25.4f} | {tfidf['mrr']:<20.4f}")
    print(f"{'Token F1 Score':<25} | {dense['token_f1']:<25.4f} | {tfidf['token_f1']:<20.4f}")
    print(f"{'Citation Accuracy':<25} | {dense['citation_accuracy']:<25.4f} | {tfidf['citation_accuracy']:<20.4f}")
    print(f"{'Abstention Accuracy':<25} | {dense['abstention_accuracy']:<25.4f} | {tfidf['abstention_accuracy']:<20.4f}")
    print(f"{'Avg Retrieval Latency (ms)':<25} | {dense['avg_retrieval_latency_ms']:<25.2f} | {tfidf['avg_retrieval_latency_ms']:<20.2f}")
    print(f"{'Avg Total Latency (ms)':<25} | {dense['avg_total_latency_ms']:<25.2f} | {tfidf['avg_total_latency_ms']:<20.2f}")
    print("-" * 75)

    print("\nGenerating EDA and Comparison Visualizations...")
    generate_all_charts()
    print("Evaluation successfully completed. Results saved to results/.")


if __name__ == "__main__":
    main()
