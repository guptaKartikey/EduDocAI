"""Dense semantic retrieval module using Sentence Transformers and FAISS / Cosine Similarity."""

import time
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass, asdict
import numpy as np
import faiss

from config.settings import settings
from src.preprocessing.chunking import DocumentChunk
from src.embeddings.embedding_model import EmbeddingModel
from src.retrieval.document_filter import DocumentFilter
from src.utils.logger import logger


@dataclass
class SearchResult:
    """Represents a retrieved document chunk with similarity score and provenance."""
    chunk: DocumentChunk
    score: float
    retrieval_method: str
    rank: int

    def to_dict(self) -> Dict[str, Any]:
        """Serializes result into dictionary."""
        return {
            "chunk_id": self.chunk.chunk_id,
            "doc_id": self.chunk.doc_id,
            "source": self.chunk.source,
            "page": self.chunk.page,
            "chunk_text": self.chunk.chunk_text,
            "score": float(self.score),
            "retrieval_method": self.retrieval_method,
            "rank": self.rank
        }


class DenseRetriever:
    """Dense vector retriever using embeddings and FAISS index with pre-filtering support."""

    def __init__(
        self,
        embedding_model: Optional[EmbeddingModel] = None,
        top_k: int = settings.DEFAULT_TOP_K
    ):
        self.embedding_model = embedding_model or EmbeddingModel.get_instance()
        self.top_k = top_k
        self.chunks: List[DocumentChunk] = []
        self.embeddings: Optional[np.ndarray] = None
        self.index: Optional[faiss.IndexFlatIP] = None

    def build_index(self, chunks: List[DocumentChunk]) -> None:
        """Encodes all document chunks and builds a FAISS inner-product index.

        Args:
            chunks: List of DocumentChunk objects.
        """
        if not chunks:
            self.chunks = []
            self.embeddings = None
            self.index = None
            logger.warning("DenseRetriever: Empty chunk list provided for index building.")
            return

        self.chunks = list(chunks)
        texts = [c.chunk_text for c in self.chunks]

        logger.info(f"DenseRetriever: Encoding {len(texts)} chunks...")
        t0 = time.perf_counter()
        self.embeddings = self.embedding_model.encode(texts, normalize=True)
        build_time = time.perf_counter() - t0

        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(self.embeddings)
        logger.info(f"DenseRetriever: FAISS index constructed in {build_time:.3f}s with {len(self.chunks)} vectors.")

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        selected_sources: Optional[List[str]] = None,
        selected_doc_ids: Optional[List[str]] = None
    ) -> List[SearchResult]:
        """Retrieves top-K most semantically similar chunks with pre-filtering.

        Args:
            query: User's question.
            top_k: Number of chunks to retrieve (defaults to self.top_k).
            selected_sources: Allowed document filenames (empty = search all).
            selected_doc_ids: Allowed document IDs (empty = search all).

        Returns:
            List[SearchResult]: Ranked retrieved results.
        """
        k = top_k if top_k is not None else self.top_k
        if not query or not query.strip() or not self.chunks or self.embeddings is None:
            return []

        # Document-level pre-filtering (Master Prompt Section 7)
        # Determine candidate indices that match selected documents
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
            logger.info("DenseRetriever: No chunks match selected document filters.")
            return []

        # Encode query to normalized vector
        query_vec = self.embedding_model.encode(query.strip(), normalize=True)  # Shape (1, dim)

        # Pre-filtered dense similarity search
        filtered_embeddings = self.embeddings[candidate_indices]  # Shape (M, dim)
        # Dot product on normalized vectors equals cosine similarity
        similarities = np.dot(filtered_embeddings, query_vec.T).squeeze(axis=1)  # Shape (M,)

        # Rank indices in descending order
        num_results = min(k, len(candidate_indices))
        if num_results <= 0:
            return []

        # Top-K indices within filtered subset
        sorted_sub_indices = np.argsort(-similarities)[:num_results]

        results: List[SearchResult] = []
        for rank, sub_idx in enumerate(sorted_sub_indices, start=1):
            original_idx = candidate_indices[sub_idx]
            score = float(np.clip(similarities[sub_idx], -1.0, 1.0))
            results.append(
                SearchResult(
                    chunk=self.chunks[original_idx],
                    score=score,
                    retrieval_method="dense",
                    rank=rank
                )
            )

        return results
