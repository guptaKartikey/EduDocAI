"""Configuration settings for EduDocAI."""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List

# Base Project Directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SAMPLE_DOCS_DIR = DATA_DIR / "sample_documents"
EVAL_DIR = DATA_DIR / "evaluation"
VECTOR_STORE_DIR = BASE_DIR / "vector_store"
RESULTS_DIR = BASE_DIR / "results"
CHARTS_DIR = RESULTS_DIR / "charts"
METRICS_DIR = RESULTS_DIR / "metrics"
REPORTS_DIR = BASE_DIR / "reports"

# Ensure all primary runtime directories exist
for directory in [DATA_DIR, SAMPLE_DOCS_DIR, EVAL_DIR, VECTOR_STORE_DIR, RESULTS_DIR, CHARTS_DIR, METRICS_DIR, REPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


@dataclass
class AppConfig:
    """Master application configuration with sensible defaults."""
    # Project Identity
    APP_NAME: str = "EduDocAI — Intelligent Document Assistant"
    APP_VERSION: str = "2.0.0"
    APP_SUBTITLE: str = "Track C: NLP / Generative AI — RAG-Powered Document Q&A System"

    # Ingestion & Documents
    ALLOWED_EXTENSIONS: List[str] = field(default_factory=lambda: [".pdf", ".txt", ".md"])
    MAX_FILE_SIZE_MB: int = 25

    # Intelligent Chunking
    DEFAULT_CHUNK_SIZE: int = 400      # in words/tokens
    DEFAULT_CHUNK_OVERLAP: int = 60    # in words/tokens
    MIN_CHUNK_SIZE: int = 50
    MAX_CHUNK_SIZE: int = 1500

    # Embeddings & Dense Retrieval
    DEFAULT_EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_BATCH_SIZE: int = 32
    NORMALIZE_EMBEDDINGS: bool = True

    # Retrieval
    DEFAULT_TOP_K: int = 3
    MAX_TOP_K: int = 10
    MIN_SIMILARITY_THRESHOLD: float = 0.02

    # LLM Providers (OpenAI-compatible + Ollama)
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_DEFAULT_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3:8b")
    OLLAMA_TIMEOUT_SECONDS: int = 45
    LLM_TEMPERATURE: float = 0.0

    # Reproducibility
    RANDOM_SEED: int = 42

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # Strict System Prompt as required by specification
    SYSTEM_PROMPT: str = (
        "You are EduDocAI, a document-grounded question-answering assistant.\n\n"
        "Answer the user's question ONLY using the supplied retrieved document context.\n\n"
        "Rules:\n"
        "1. Do not use outside knowledge.\n"
        "2. Do not invent facts, numbers, names, dates, or policies.\n"
        "3. If the answer is not supported by the retrieved context, explicitly say:\n"
        "   'The requested information was not found in the uploaded documents.'\n"
        "4. Do not combine unrelated documents unless the question requires it.\n"
        "5. Preserve important numbers and terminology exactly.\n"
        "6. Cite the source document and page number for every factual claim.\n"
        "7. If sources conflict, explain the conflict and cite both sources.\n"
        "8. Distinguish between directly stated information and reasonable interpretation.\n"
        "9. Never fabricate citations.\n"
        "10. If the retrieved context is insufficient, abstain instead of guessing."
    )

    ABSTENTION_MESSAGE: str = "The requested information was not found in the uploaded documents."


# Global settings instance
settings = AppConfig()
