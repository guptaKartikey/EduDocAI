"""Embedding model wrapper for EduDocAI using Sentence Transformers."""

from typing import List, Optional, Union
import numpy as np

from sentence_transformers import SentenceTransformer
from config.settings import settings
from src.utils.logger import logger


class EmbeddingModel:
    """Manages sentence embedding generation with batching and normalization."""

    _instance: Optional["EmbeddingModel"] = None
    _loaded_model_name: Optional[str] = None
    _model: Optional[SentenceTransformer] = None

    def __init__(self, model_name: str = settings.DEFAULT_EMBEDDING_MODEL):
        self.model_name = model_name
        self._load_model()

    @classmethod
    def get_instance(cls, model_name: str = settings.DEFAULT_EMBEDDING_MODEL) -> "EmbeddingModel":
        """Singleton accessor to prevent reloading weights into memory repeatedly."""
        if cls._instance is None or cls._loaded_model_name != model_name:
            cls._instance = cls(model_name)
            cls._loaded_model_name = model_name
        return cls._instance

    def _load_model(self) -> None:
        """Loads the SentenceTransformer model weights."""
        try:
            logger.info(f"Loading embedding model '{self.model_name}'...")
            self._model = SentenceTransformer(self.model_name)
            logger.info(f"Model '{self.model_name}' loaded successfully. Embedding dimension: {self.dimension}")
        except Exception as e:
            logger.error(f"Failed to load embedding model '{self.model_name}': {str(e)}")
            raise RuntimeError(
                f"Could not load SentenceTransformer '{self.model_name}'. "
                "Ensure internet access for initial download or verify local cache."
            ) from e

    @property
    def dimension(self) -> int:
        """Returns the embedding vector dimension."""
        if self._model is None:
            raise RuntimeError("Model is not loaded.")
        if hasattr(self._model, "get_embedding_dimension"):
            dim = self._model.get_embedding_dimension()
        else:
            dim = self._model.get_sentence_embedding_dimension()
        return int(dim) if dim is not None else 384

    def encode(
        self,
        texts: Union[str, List[str]],
        normalize: bool = settings.NORMALIZE_EMBEDDINGS,
        batch_size: int = settings.EMBEDDING_BATCH_SIZE
    ) -> np.ndarray:
        """Encodes texts into L2-normalized float32 vectors.

        Args:
            texts: Single string or list of text strings.
            normalize: Whether to L2-normalize vectors for cosine similarity.
            batch_size: Mini-batch size for inference.

        Returns:
            np.ndarray: Matrix of shape (N, dimension) or (1, dimension).
        """
        if self._model is None:
            raise RuntimeError("Embedding model is not initialized.")

        if isinstance(texts, str):
            texts = [texts]

        if len(texts) == 0:
            return np.empty((0, self.dimension), dtype=np.float32)

        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=normalize,
            convert_to_numpy=True
        )

        return embeddings.astype(np.float32)
