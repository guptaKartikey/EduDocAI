"""TF-IDF sparse retrieval module using Scikit-Learn."""

import time
from typing import List, Optional, Dict, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config.settings import settings
from src.preprocessing.chunking import DocumentChunk
from src.retrieval.dense_retriever import SearchResult
from src.utils.logger import logger


class TfidfRetriever:
    """Sparse retriever using Term Frequency-Inverse Document Frequency (TF-IDF)."""

    def __init__(self, top_k: int = settings.DEFAULT_TOP_K):
        self.top_k = top_k
        self.chunks: List[DocumentChunk] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None

    def build_index(self, chunks: List[DocumentChunk]) -> None:
        """Fits TF-IDF vectorizer and builds sparse matrix across all chunks.

        Args:
            chunks: Collection of DocumentChunk objects.
        """
        if not chunks:
            self.chunks = []
            self.vectorizer = None
            self.tfidf_matrix = None
            logger.warning("TfidfRetriever: Empty chunk list provided for index building.")
            return

        self.chunks = list(chunks)
        texts = [c.chunk_text for c in self.chunks]

        t0 = time.perf_counter()
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        build_time = time.perf_counter() - t0

        logger.info(
            f"TfidfRetriever: Index built in {build_time:.3f}s. Vocabulary size: {len(self.vectorizer.vocabulary_)} terms."
        )

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        selected_sources: Optional[List[str]] = None,
        selected_doc_ids: Optional[List[str]] = None
    ) -> List[SearchResult]:
        """Performs TF-IDF query matching with pre-filtering support.

        Args:
            query: User's question.
            top_k: Number of chunks to retrieve.
            selected_sources: Allowed document filenames.
            selected_doc_ids: Allowed document IDs.

        Returns:
            List[SearchResult]: Ranked retrieved results with cosine similarity scores.
        """
        k = top_k if top_k is not None else self.top_k
        if not query or not query.strip() or not self.chunks or self.vectorizer is None or self.tfidf_matrix is None:
            return []

        # Document-level pre-filtering (Master Prompt Section 7)
        candidate_indices = []
        for idx, chunk in enumerate(self.chunks):
            match_source = chunk.source in selected_sources if selected_sources else True
            match_id = chunk.doc_id in selected_doc_ids if selected_doc_ids else True

            if (selected_sources and not selected_doc_ids and match_source) or \
               (selected_doc_ids and not selected_sources and match_id) or \
               (selected_sources and selected_doc_ids and (match_source or match_id)) or \
               (not selected_sources and not selected_doc_ids):
                candidate_indices.append(idx)

        if not candidate_indices:
            logger.info("TfidfRetriever: No chunks match selected document filters.")
            return []

        # Transform query to TF-IDF vector
        query_vec = self.vectorizer.transform([query.strip()])

        # Pre-filtered TF-IDF similarity calculation
        filtered_matrix = self.tfidf_matrix[candidate_indices]
        similarities = cosine_similarity(filtered_matrix, query_vec).flatten()

        num_results = min(k, len(candidate_indices))
        if num_results <= 0:
            return []

        # Sort descending
        sorted_sub_indices = np.argsort(-similarities)[:num_results]

        results: List[SearchResult] = []
        for rank, sub_idx in enumerate(sorted_sub_indices, start=1):
            original_idx = candidate_indices[sub_idx]
            score = float(np.clip(similarities[sub_idx], 0.0, 1.0))
            results.append(
                SearchResult(
                    chunk=self.chunks[original_idx],
                    score=score,
                    retrieval_method="tfidf",
                    rank=rank
                )
            )

        return results
