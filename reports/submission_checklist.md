# EduRankAI: Internship Submission & Verification Checklist

**Candidate:** Kartikey Gupta  
**Assignment Track:** Track C: NLP / Generative AI  
**Assignment Role:** AI Engineering & Data Science Intern  
**Project Name:** EduRankAI (Track C: Intelligent Document Intelligence System)  
**Submission Date:** November 2026  

---

## 📋 Comprehensive Deliverables Checklist

### 1. Source Code & Architecture
- [x] Complete, fully executable Python codebase (no pseudocode or TODO stubs).
- [x] Clean modular directory structure matching specification (`src/ingestion`, `src/preprocessing`, `src/embeddings`, `src/retrieval`, `src/generation`, `src/evaluation`, `src/utils`).
- [x] `requirements.txt` with all dependencies pinned.
- [x] `.env.example` and `.gitignore` included.
- [x] `config/settings.py` with typed configurations, system prompts, paths, and reproducibility seeds.

### 2. Document Ingestion & In-Memory PDF Processing
- [x] Support for PDF, TXT, and Markdown files.
- [x] PyMuPDF (`fitz`) in-memory parsing using `io.BytesIO` to prevent Windows file-locking.
- [x] Granular page-by-page metadata preservation (`doc_id`, `source`, `page`, `text`).
- [x] Graceful handling of encrypted PDFs, corrupted streams, and empty documents.
- [x] Informative reporting for image-only scans requiring OCR.

### 3. Intelligent Chunking
- [x] Configurable chunk size (default: 400 words) and sliding-window overlap (default: 60 words).
- [x] Sentence-boundary awareness via regex heuristics.
- [x] Comprehensive metadata preservation on every chunk (`chunk_id`, `doc_id`, `source`, `page`, `word_count`).
- [x] Documented architectural explanation of parameter trade-offs.

### 4. Dual Retrieval & Document-Level Isolation
- [x] Dense semantic retrieval with Sentence-Transformers (`all-MiniLM-L6-v2`) and FAISS inner-product vector indexing (`IndexFlatIP`).
- [x] Sparse lexical retrieval using Scikit-Learn `TfidfVectorizer` baseline with sublinear scaling.
- [x] Hybrid Reciprocal Rank Fusion (RRF) reranker ($k=60$).
- [x] **Strict Pre-Retrieval Document Filtering**: Search across all, single, or multiple selected documents with zero cross-document contamination.
- [x] Transparent score reporting without false percentage claims.

### 5. Grounded Generation & Safety Gate
- [x] Local LLM integration via Ollama client with timeout and health-check safeguards.
- [x] High-precision Extractive Grounded Fallback QA engine when Ollama is offline or uninstalled.
- [x] Strict Grounding Validator that audits token-level claim overlap and citation authenticity.
- [x] Deterministic abstention message when evidence is insufficient:
  *"The requested information was not found in the uploaded documents."*
- [x] Verifiable citations displaying source document, page number, rank, and similarity score.

### 6. Interactive User Interface (Single-Flow Unified Screen)
- [x] Modern Streamlit single-screen dashboard (`app.py`):
  - User uploads document $\rightarrow$ Asks Q&A $\rightarrow$ Generates Grounded Answer, Citations, Executive Summary, Text Analytics, Domain Classification, and Retrieval Benchmark on one screen.
- [x] Sidebar controls for uploading files, one-click demo loading, and document target selection.
- [x] Expandable evidence inspection viewer.

### 7. Benchmark Dataset & Quantitative Evaluation
- [x] Curated 22-question benchmark dataset (`data/evaluation/eval_dataset.json`) spanning 6 categories (factual, numeric, paraphrased, multi-document, ambiguous, abstention).
- [x] Evaluation metrics implemented: Recall@1, Recall@3, Recall@5, MRR, Token F1, Exact Match, Citation Accuracy, Abstention Accuracy, and Latencies.
- [x] Executable evaluation runner (`scripts/evaluate.py`).
- [x] Evaluation results calculated from actual execution and saved to `results/metrics/`.

### 8. Exploratory Data Analysis (EDA) Visualizations
- [x] `01_document_sources.png`: Page count inventory by source document.
- [x] `02_chunk_length_distribution.png`: Word count distribution across text chunks.
- [x] `03_eval_question_categories.png`: Benchmark question distribution by category.
- [x] `04_dense_vs_tfidf_comparison.png`: Comprehensive performance comparison between Dense and Baseline Sparse retrieval.

### 9. Test Suite & Verification
- [x] 32 automated unit and integration tests using Pytest.
- [x] 100% pass rate verified on Windows 10/11 (`32 passed`).
- [x] Tests cover ingestion, corrupted files, chunking, dense/sparse search, filtering isolation, generation, and metrics.

### 10. Documentation & Reports
- [x] Comprehensive `README.md` with Windows PowerShell setup commands and architecture diagrams.
- [x] In-depth technical report (`reports/technical_report.md`) matching the 7-section EduRankAI structure.
- [x] Structured 3–5 minute evaluation demo script (`reports/demo_script.md`).
- [x] Demonstration synthetic datasets provided in `data/sample_documents/` with explicit non-confidential disclaimers.

---

## 📝 Candidate Declaration (Section 10)

I confirm that the submitted work accurately represents my contribution. I have identified external sources and disclosed AI tools or third-party code. I understand that I may be asked to explain or reproduce key parts of the implementation.

**Applicant Name:** Kartikey Gupta  
**Role:** AI Engineering & Data Science Intern  
**Assignment Track:** Track C: NLP / Generative AI  
**Date:** November 2026  
**Status:** Completed & Verified  
