"""Automated RAG evaluation pipeline for EduDocAI."""

import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd

from config.settings import settings, EVAL_DIR, SAMPLE_DOCS_DIR, METRICS_DIR
from src.ingestion.document_loader import DocumentLoader
from src.preprocessing.chunking import TextChunker
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.tfidf_retriever import TfidfRetriever
from src.generation.answer_generator import AnswerGenerator
from src.evaluation.metrics import RetrievalMetrics, GenerationMetrics
from src.utils.logger import logger


class RAGEvaluator:
    """Executes reproducible quantitative evaluation of retrieval and generation."""

    def __init__(
        self,
        eval_dataset_path: Path = EVAL_DIR / "eval_dataset.json",
        sample_docs_dir: Path = SAMPLE_DOCS_DIR
    ):
        self.eval_dataset_path = eval_dataset_path
        self.sample_docs_dir = sample_docs_dir

        self.loader = DocumentLoader()
        self.chunker = TextChunker()
        self.dense_retriever = DenseRetriever()
        self.tfidf_retriever = TfidfRetriever()
        self.answer_generator = AnswerGenerator()

        self.eval_data: List[Dict[str, Any]] = []
        self.chunks = []

    def load_dataset(self) -> List[Dict[str, Any]]:
        """Loads evaluation test cases from JSON."""
        if not self.eval_dataset_path.exists():
            raise FileNotFoundError(f"Evaluation dataset not found at {self.eval_dataset_path}")

        with open(self.eval_dataset_path, "r", encoding="utf-8") as f:
            self.eval_data = json.load(f)

        logger.info(f"Loaded {len(self.eval_data)} test questions from {self.eval_dataset_path}")
        return self.eval_data

    def index_sample_documents(self) -> int:
        """Loads and chunks all sample documents for benchmarking."""
        # Find all sample PDFs
        pdf_files = list(self.sample_docs_dir.glob("*.pdf"))
        if not pdf_files:
            # Fallback to TXT
            pdf_files = list(self.sample_docs_dir.glob("*.txt"))

        all_pages = []
        t0 = time.perf_counter()
        for p in pdf_files:
            pages = self.loader.load_from_path(p)
            all_pages.extend(pages)

        self.chunks = self.chunker.chunk_documents(all_pages)
        self.dense_retriever.build_index(self.chunks)
        self.tfidf_retriever.build_index(self.chunks)
        index_time = time.perf_counter() - t0

        logger.info(f"Indexed {len(all_pages)} pages into {len(self.chunks)} chunks in {index_time:.3f}s.")
        return len(self.chunks)

    def run_evaluation(self, top_k: int = 3, use_ollama: bool = False) -> Dict[str, Any]:
        """Runs end-to-end evaluation comparing Dense vs TF-IDF on actual test cases.

        Args:
            top_k: Number of retrieved chunks to consider.
            use_ollama: Whether to attempt Ollama for generation (defaults to False for fast reproducibility).

        Returns:
            Dict[str, Any]: Comprehensive metrics dictionary.
        """
        if not self.eval_data:
            self.load_dataset()
        if not self.chunks:
            self.index_sample_documents()

        results_dense = []
        results_tfidf = []

        total_dense_retrieval_time = 0.0
        total_tfidf_retrieval_time = 0.0
        total_dense_gen_time = 0.0
        total_tfidf_gen_time = 0.0

        for item in self.eval_data:
            q = item["question"]
            target_doc = item.get("relevant_document")
            target_page = item.get("relevant_page")
            expected_ans = item["expected_answer"]
            is_abstention = item.get("is_abstention", False)

            # 1. Dense Retrieval Run
            t0 = time.perf_counter()
            dense_chunks = self.dense_retriever.search(q, top_k=5)
            dense_ret_latency = time.perf_counter() - t0
            total_dense_retrieval_time += dense_ret_latency

            t1 = time.perf_counter()
            dense_answer_obj = self.answer_generator.generate_answer(q, dense_chunks[:top_k], use_ollama=use_ollama)
            dense_gen_latency = time.perf_counter() - t1
            total_dense_gen_time += dense_gen_latency

            dense_chunk_dicts = [r.to_dict() for r in dense_chunks]
            r1_dense = RetrievalMetrics.recall_at_k(dense_chunk_dicts, target_doc, target_page, k=1) if not is_abstention else 1.0
            r3_dense = RetrievalMetrics.recall_at_k(dense_chunk_dicts, target_doc, target_page, k=3) if not is_abstention else 1.0
            r5_dense = RetrievalMetrics.recall_at_k(dense_chunk_dicts, target_doc, target_page, k=5) if not is_abstention else 1.0
            mrr_dense = RetrievalMetrics.reciprocal_rank(dense_chunk_dicts, target_doc, target_page) if not is_abstention else 1.0

            dense_f1 = GenerationMetrics.token_f1(dense_answer_obj.answer, expected_ans)
            dense_em = GenerationMetrics.exact_match(dense_answer_obj.answer, expected_ans)
            dense_cite = GenerationMetrics.citation_accuracy(dense_answer_obj.citations, target_doc, target_page)
            dense_abs = GenerationMetrics.abstention_accuracy(dense_answer_obj.is_abstained, is_abstention)

            results_dense.append({
                "id": item["id"],
                "category": item["category"],
                "question": q,
                "recall@1": r1_dense,
                "recall@3": r3_dense,
                "recall@5": r5_dense,
                "mrr": mrr_dense,
                "f1": dense_f1,
                "exact_match": dense_em,
                "citation_accuracy": dense_cite,
                "abstention_accuracy": dense_abs,
                "groundedness": dense_answer_obj.groundedness_score,
                "retrieval_latency_ms": dense_ret_latency * 1000.0,
                "total_latency_ms": (dense_ret_latency + dense_gen_latency) * 1000.0,
                "is_abstained": dense_answer_obj.is_abstained
            })

            # 2. TF-IDF Retrieval Run
            t0 = time.perf_counter()
            tfidf_chunks = self.tfidf_retriever.search(q, top_k=5)
            tfidf_ret_latency = time.perf_counter() - t0
            total_tfidf_retrieval_time += tfidf_ret_latency

            t1 = time.perf_counter()
            tfidf_answer_obj = self.answer_generator.generate_answer(q, tfidf_chunks[:top_k], use_ollama=use_ollama)
            tfidf_gen_latency = time.perf_counter() - t1
            total_tfidf_gen_time += tfidf_gen_latency

            tfidf_chunk_dicts = [r.to_dict() for r in tfidf_chunks]
            r1_tfidf = RetrievalMetrics.recall_at_k(tfidf_chunk_dicts, target_doc, target_page, k=1) if not is_abstention else 1.0
            r3_tfidf = RetrievalMetrics.recall_at_k(tfidf_chunk_dicts, target_doc, target_page, k=3) if not is_abstention else 1.0
            r5_tfidf = RetrievalMetrics.recall_at_k(tfidf_chunk_dicts, target_doc, target_page, k=5) if not is_abstention else 1.0
            mrr_tfidf = RetrievalMetrics.reciprocal_rank(tfidf_chunk_dicts, target_doc, target_page) if not is_abstention else 1.0

            tfidf_f1 = GenerationMetrics.token_f1(tfidf_answer_obj.answer, expected_ans)
            tfidf_em = GenerationMetrics.exact_match(tfidf_answer_obj.answer, expected_ans)
            tfidf_cite = GenerationMetrics.citation_accuracy(tfidf_answer_obj.citations, target_doc, target_page)
            tfidf_abs = GenerationMetrics.abstention_accuracy(tfidf_answer_obj.is_abstained, is_abstention)

            results_tfidf.append({
                "id": item["id"],
                "category": item["category"],
                "question": q,
                "recall@1": r1_tfidf,
                "recall@3": r3_tfidf,
                "recall@5": r5_tfidf,
                "mrr": mrr_tfidf,
                "f1": tfidf_f1,
                "exact_match": tfidf_em,
                "citation_accuracy": tfidf_cite,
                "abstention_accuracy": tfidf_abs,
                "groundedness": tfidf_answer_obj.groundedness_score,
                "retrieval_latency_ms": tfidf_ret_latency * 1000.0,
                "total_latency_ms": (tfidf_ret_latency + tfidf_gen_latency) * 1000.0,
                "is_abstained": tfidf_answer_obj.is_abstained
            })

        df_dense = pd.DataFrame(results_dense)
        df_tfidf = pd.DataFrame(results_tfidf)

        n = len(self.eval_data)
        summary = {
            "dataset_size": n,
            "dense_retrieval": {
                "mean_recall@1": float(df_dense["recall@1"].mean()),
                "mean_recall@3": float(df_dense["recall@3"].mean()),
                "mean_recall@5": float(df_dense["recall@5"].mean()),
                "mrr": float(df_dense["mrr"].mean()),
                "token_f1": float(df_dense["f1"].mean()),
                "citation_accuracy": float(df_dense["citation_accuracy"].mean()),
                "abstention_accuracy": float(df_dense["abstention_accuracy"].mean()),
                "avg_groundedness": float(df_dense["groundedness"].mean()),
                "avg_retrieval_latency_ms": float(df_dense["retrieval_latency_ms"].mean()),
                "avg_total_latency_ms": float(df_dense["total_latency_ms"].mean())
            },
            "tfidf_retrieval": {
                "mean_recall@1": float(df_tfidf["recall@1"].mean()),
                "mean_recall@3": float(df_tfidf["recall@3"].mean()),
                "mean_recall@5": float(df_tfidf["recall@5"].mean()),
                "mrr": float(df_tfidf["mrr"].mean()),
                "token_f1": float(df_tfidf["f1"].mean()),
                "citation_accuracy": float(df_tfidf["citation_accuracy"].mean()),
                "abstention_accuracy": float(df_tfidf["abstention_accuracy"].mean()),
                "avg_groundedness": float(df_tfidf["groundedness"].mean()),
                "avg_retrieval_latency_ms": float(df_tfidf["retrieval_latency_ms"].mean()),
                "avg_total_latency_ms": float(df_tfidf["total_latency_ms"].mean())
            },
            "category_breakdown": df_dense.groupby("category")[["recall@3", "f1", "abstention_accuracy"]].mean().to_dict(orient="index")
        }

        # Save summary JSON
        out_file = METRICS_DIR / "evaluation_summary.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        # Save detailed CSVs
        df_dense.to_csv(METRICS_DIR / "dense_evaluation_details.csv", index=False)
        df_tfidf.to_csv(METRICS_DIR / "tfidf_evaluation_details.csv", index=False)

        logger.info(f"Evaluation complete. Summary saved to {out_file}")
        return summary
