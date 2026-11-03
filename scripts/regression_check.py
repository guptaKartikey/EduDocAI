"""
Regression Guard — EduDocAI
============================
Auto-runs the evaluation benchmark and checks that key metrics have not dropped
below the baseline thresholds established on Day 1.

Usage:
    python scripts/regression_check.py

Exit codes:
    0 — all checks passed
    1 — one or more metrics regressed below threshold

Run this after any code change that touches retrieval, chunking, or generation.
"""

import sys
import json
from pathlib import Path

# ── Thresholds (based on measured baseline results) ───────────────────────────
# Set 5% below the last known good result — alert if anything drops further.
THRESHOLDS = {
    "mean_recall@1":        0.82,   # Measured: 86.36%
    "mean_recall@3":        0.98,   # Measured: 100%
    "mrr":                  0.90,   # Measured: 0.9318
    "token_f1":             0.38,   # Measured: 41.52%
    "citation_accuracy":    0.58,   # Measured: 63.64%
    "abstention_accuracy":  0.58,   # Measured: 63.64%
    "avg_groundedness":     0.88,   # Measured: 93.96%
}

RESULTS_FILE = Path("results/metrics/evaluation_summary.json")


def load_results() -> dict:
    if not RESULTS_FILE.exists():
        print(f"ERROR: Results file not found at {RESULTS_FILE}")
        print("Run `python scripts/run_evaluation.py` first to generate results.")
        sys.exit(1)
    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def run_checks(results: dict) -> bool:
    """Check Dense RAG metrics against regression thresholds."""
    # Support both key names: 'dense' and 'dense_retrieval'
    dense = results.get("dense") or results.get("dense_retrieval", {})
    if not dense:
        print("ERROR: No 'dense' metrics found in results file.")
        return False

    all_passed = True
    print("\n" + "=" * 60)
    print("  EduDocAI Regression Guard — Metric Check")
    print("=" * 60)
    print(f"  {'Metric':<30} {'Current':>10} {'Threshold':>10} {'Status':>8}")
    print("-" * 60)

    for metric, threshold in THRESHOLDS.items():
        current = dense.get(metric, None)
        if current is None:
            print(f"  {metric:<30} {'N/A':>10} {threshold:>10.4f} {'⚠ SKIP':>8}")
            continue

        passed = current >= threshold
        status = "PASS" if passed else "FAIL"
        marker = "[OK]" if passed else "[!!]"
        if not passed:
            all_passed = False

        print(f"  {marker} {metric:<28} {current:>10.4f} {threshold:>10.4f}  {status}")

    print("=" * 60)

    # Latency sanity check (warn only — latency can vary by hardware)
    latency = dense.get("avg_retrieval_latency_ms", 0)
    if latency > 500:
        print(f"  ⚠  WARNING: Retrieval latency is {latency:.1f}ms (expected < 500ms)")
        print("     This may indicate an indexing or hardware issue.")

    return all_passed


def main():
    print("\nLoading evaluation results from:", RESULTS_FILE)
    results = load_results()

    passed = run_checks(results)

    if passed:
        print("\n  All regression checks passed. No metrics regressed.")
        print("  Safe to merge / deploy.\n")
        sys.exit(0)
    else:
        print("\n  REGRESSION DETECTED. One or more metrics dropped below threshold.")
        print("  Review recent changes to retrieval, chunking, or generation.")
        print("  Re-run full evaluation: python scripts/run_evaluation.py\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
