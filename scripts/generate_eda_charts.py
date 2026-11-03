"""Generates publication-quality EDA and benchmark comparison visualizations."""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from config.settings import CHARTS_DIR, METRICS_DIR, EVAL_DIR, SAMPLE_DOCS_DIR
from src.ingestion.document_loader import DocumentLoader
from src.preprocessing.chunking import TextChunker
from src.evaluation.evaluator import RAGEvaluator
from src.utils.logger import logger

# Set aesthetic styling
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "figure.titlesize": 16
})


def generate_all_charts():
    """Generates and saves the 4 required EDA and benchmark visualizations."""
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    loader = DocumentLoader()
    chunker = TextChunker()

    # Load sample documents
    sample_files = list(SAMPLE_DOCS_DIR.glob("*.pdf"))
    if not sample_files:
        sample_files = list(SAMPLE_DOCS_DIR.glob("*.txt"))

    all_pages = []
    for f in sample_files:
        all_pages.extend(loader.load_from_path(f))

    chunks = chunker.chunk_documents(all_pages)

    # 1. Document Count & Pages by Source
    doc_stats = {}
    for p in all_pages:
        doc_stats[p.source] = doc_stats.get(p.source, 0) + 1

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sources = [s.replace(".pdf", "") for s in doc_stats.keys()]
    counts = list(doc_stats.values())
    bars = ax.barh(sources, counts, color="#2b5c8f", height=0.55)
    ax.set_xlabel("Number of Pages")
    ax.set_title("1. Document Inventory: Page Count by Source Document")
    for bar in bars:
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2, f"{int(bar.get_width())} pages", va="center")
    plt.tight_layout()
    chart1_path = CHARTS_DIR / "01_document_sources.png"
    plt.savefig(chart1_path, dpi=300)
    plt.close()
    logger.info(f"Saved Chart 1: {chart1_path}")

    # 2. Chunk Length Distribution
    lengths = [c.word_count for c in chunks]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.histplot(lengths, bins=10, kde=True, color="#1e88e5", ax=ax)
    mean_val = np.mean(lengths)
    median_val = np.median(lengths)
    ax.axvline(mean_val, color="#d32f2f", linestyle="--", label=f"Mean ({mean_val:.1f} words)")
    ax.axvline(median_val, color="#388e3c", linestyle="-.", label=f"Median ({median_val:.1f} words)")
    ax.set_xlabel("Chunk Length (Words)")
    ax.set_ylabel("Frequency")
    ax.set_title("2. Intelligent Chunking: Word Count Distribution")
    ax.legend()
    plt.tight_layout()
    chart2_path = CHARTS_DIR / "02_chunk_length_distribution.png"
    plt.savefig(chart2_path, dpi=300)
    plt.close()
    logger.info(f"Saved Chart 2: {chart2_path}")

    # 3. Evaluation Question Distribution by Category
    eval_file = EVAL_DIR / "eval_dataset.json"
    with open(eval_file, "r", encoding="utf-8") as f:
        eval_cases = json.load(f)

    cat_counts = {}
    for c in eval_cases:
        cat = c["category"]
        cat_counts[cat] = cat_counts.get(cat, 0) + 1

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors_list = ["#2b5c8f", "#43a047", "#f4511e", "#8e24aa", "#00acc1", "#fbc02d"]
    bars = ax.bar(cat_counts.keys(), cat_counts.values(), color=colors_list[:len(cat_counts)], width=0.6)
    ax.set_ylabel("Number of Test Cases")
    ax.set_title("3. Evaluation Benchmark: Question Distribution by Category")
    plt.xticks(rotation=20, ha="right")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, f"{int(bar.get_height())}", ha="center")
    plt.tight_layout()
    chart3_path = CHARTS_DIR / "03_eval_question_categories.png"
    plt.savefig(chart3_path, dpi=300)
    plt.close()
    logger.info(f"Saved Chart 3: {chart3_path}")

    # 4. Dense vs TF-IDF Retrieval Comparison
    metrics_file = METRICS_DIR / "evaluation_summary.json"
    if not metrics_file.exists():
        evaluator = RAGEvaluator()
        metrics = evaluator.run_evaluation(top_k=3, use_ollama=False)
    else:
        with open(metrics_file, "r", encoding="utf-8") as f:
            metrics = json.load(f)

    dense_m = metrics["dense_retrieval"]
    tfidf_m = metrics["tfidf_retrieval"]

    metric_names = ["Recall@1", "Recall@3", "Recall@5", "MRR", "Token F1", "Abstention Acc."]
    dense_vals = [
        dense_m["mean_recall@1"],
        dense_m["mean_recall@3"],
        dense_m["mean_recall@5"],
        dense_m["mrr"],
        dense_m["token_f1"],
        dense_m["abstention_accuracy"]
    ]
    tfidf_vals = [
        tfidf_m["mean_recall@1"],
        tfidf_m["mean_recall@3"],
        tfidf_m["mean_recall@5"],
        tfidf_m["mrr"],
        tfidf_m["token_f1"],
        tfidf_m["abstention_accuracy"]
    ]

    x = np.arange(len(metric_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    rects1 = ax.bar(x - width/2, dense_vals, width, label="Dense (all-MiniLM-L6-v2)", color="#1976d2")
    rects2 = ax.bar(x + width/2, tfidf_vals, width, label="Sparse (TF-IDF)", color="#78909c")

    ax.set_ylabel("Score (0.0 to 1.0)")
    ax.set_title("4. Retrieval Benchmark: Dense vs TF-IDF Performance Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper right")

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.2f}",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=9)

    autolabel(rects1)
    autolabel(rects2)
    plt.tight_layout()
    chart4_path = CHARTS_DIR / "04_dense_vs_tfidf_comparison.png"
    plt.savefig(chart4_path, dpi=300)
    plt.close()
    logger.info(f"Saved Chart 4: {chart4_path}")


if __name__ == "__main__":
    generate_all_charts()
    print("All 4 EDA charts successfully generated in results/charts/")
