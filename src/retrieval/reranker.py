"""Hybrid reranker combining Dense and TF-IDF retrieval via Reciprocal Rank Fusion (RRF)."""

from typing import List, Dict, Tuple
from src.retrieval.dense_retriever import SearchResult
from src.utils.logger import logger


class HybridReranker:
    """Combines semantic dense retrieval and lexical TF-IDF retrieval using RRF."""

    @staticmethod
    def fuse(
        dense_results: List[SearchResult],
        tfidf_results: List[SearchResult],
        rrf_k: int = 60,
        top_k: int = 3
    ) -> List[SearchResult]:
        """Merges two ranked lists using Reciprocal Rank Fusion: RRF_score = sum(1 / (k + rank)).

        Args:
            dense_results: Ranked results from dense retriever.
            tfidf_results: Ranked results from TF-IDF retriever.
            rrf_k: Smoothing constant (standard RRF parameter, default 60).
            top_k: Number of combined results to return.

        Returns:
            List[SearchResult]: Re-ranked unified results with combined scores.
        """
        scores: Dict[str, float] = {}
        chunk_map = {}

        # Accumulate RRF scores from Dense retrieval
        for r in dense_results:
            cid = r.chunk.chunk_id
            chunk_map[cid] = r.chunk
            scores[cid] = scores.get(cid, 0.0) + (1.0 / (rrf_k + r.rank))

        # Accumulate RRF scores from TF-IDF retrieval
        for r in tfidf_results:
            cid = r.chunk.chunk_id
            chunk_map[cid] = r.chunk
            scores[cid] = scores.get(cid, 0.0) + (1.0 / (rrf_k + r.rank))

        # Sort chunks by descending fusion score
        sorted_cids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:top_k]

        fused_results: List[SearchResult] = []
        for rank, cid in enumerate(sorted_cids, start=1):
            fused_results.append(
                SearchResult(
                    chunk=chunk_map[cid],
                    score=float(scores[cid]),
                    retrieval_method="hybrid_rrf",
                    rank=rank
                )
            )

        logger.info(f"HybridReranker fused {len(dense_results)} dense and {len(tfidf_results)} TF-IDF chunks -> {len(fused_results)} results.")
        return fused_results
