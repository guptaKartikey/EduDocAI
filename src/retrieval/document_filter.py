"""Document-level pre-filtering module for EduDocAI."""

from typing import List, Optional, Set
from src.preprocessing.chunking import DocumentChunk
from src.utils.logger import logger


class DocumentFilter:
    """Filters document chunks prior to dense or TF-IDF ranking.

    Ensures strict document isolation: when one or more documents are selected,
    chunks from unselected documents are never searched or returned.
    """

    @staticmethod
    def filter_chunks(
        chunks: List[DocumentChunk],
        selected_sources: Optional[List[str]] = None,
        selected_doc_ids: Optional[List[str]] = None
    ) -> List[DocumentChunk]:
        """Applies pre-retrieval filtering on chunks by document name or doc_id.

        Args:
            chunks: Complete collection of indexed document chunks.
            selected_sources: List of target document filenames to include.
            selected_doc_ids: List of target document IDs to include.

        Returns:
            List[DocumentChunk]: Filtered subset of chunks belonging strictly to target documents.
        """
        if not chunks:
            return []

        # If no filter applied or empty selections, search across all documents
        if not selected_sources and not selected_doc_ids:
            return chunks

        source_set: Set[str] = set(selected_sources) if selected_sources else set()
        doc_id_set: Set[str] = set(selected_doc_ids) if selected_doc_ids else set()

        filtered: List[DocumentChunk] = []
        for chunk in chunks:
            match_source = chunk.source in source_set if source_set else False
            match_id = chunk.doc_id in doc_id_set if doc_id_set else False

            # Match either source name or doc_id
            if (source_set and match_source) or (doc_id_set and match_id) or (not source_set and match_id) or (not doc_id_set and match_source):
                filtered.append(chunk)

        logger.info(
            f"Document filter applied: {len(filtered)}/{len(chunks)} chunks retained "
            f"for targets sources={list(source_set)} doc_ids={list(doc_id_set)}."
        )
        return filtered
