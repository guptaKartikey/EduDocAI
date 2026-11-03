<div align="center">
  <img src="assets/logo.png" alt="EduDocAI Logo" width="320"/>
</div>

<h1 align="center">EduDocAI — Intelligent Document Assistant</h1>

<p align="center">
  <strong>Developed by Kartikey Gupta</strong><br>
  Track C: NLP / Generative AI — AI Engineering & Data Science Internship Assessment (EduRankAI)
</p>

<p align="center">
  <a href="https://python.org"><img src="https://img.shields.io/badge/Python-3.10%2B-blue" alt="Python"/></a>
  <a href="https://streamlit.io"><img src="https://img.shields.io/badge/Streamlit-1.35%2B-red" alt="Streamlit"/></a>
  <img src="https://img.shields.io/badge/Tests-32%20passing-brightgreen" alt="Tests"/>
  <img src="https://img.shields.io/badge/Track-C%20NLP%2FGen--AI-blueviolet" alt="Track C"/>
  <img src="https://img.shields.io/badge/License-MIT-green" alt="MIT"/>
</p>

---

## 📌 Overview

**EduDocAI** is an end-to-end AI-powered document question-answering system built for the EduRankAI internship technical assessment. It lets you:

- 📄 Upload one or more **PDF documents**
- 💬 Ask **natural-language questions** about them in a ChatGPT-style interface
- ✅ Receive **context-grounded answers** with **exact page-level citations**
- 📊 Compare traditional **TF-IDF retrieval** against modern **Dense Embedding RAG**
- 🧪 Evaluate retrieval and answer quality with real metrics (Recall@K, MRR, Token F1)
- ⚡ Quick-start with **3 built-in benchmark documents** for instant evaluation

The application is fully functional end-to-end — no mocks, no fabricated results, runs entirely **offline without any paid API**.

---

## 🏗️ Architecture

```
User Question
     │
     ▼
Query Embedding            ←──── Sentence Transformers (all-MiniLM-L6-v2)
     │
     ▼
FAISS Similarity Search   ←──── Pre-built L2-normalised inner-product index
     │
     ├──── TF-IDF Baseline (scikit-learn)
     │
     ▼
Top-K Relevant Chunks (with page metadata)
     │
     ▼
Context Construction + Grounding Validation
     │
     ▼
LLM Answer Generation (Ollama local) or Extractive Zero-Hallucination Fallback
     │
     ▼
Citation Validation + Abstention Check
     │
     ▼
Final Grounded Answer + Source Citations (Document · Page)
```

---

## ✨ Features

| Feature | Details |
|---|---|
| PDF Upload | PyMuPDF, multi-file, max 25 MB each |
| Quick-Start Demo | 1-click loads 3 benchmark PDFs for instant testing |
| Chunking | Configurable sliding-window (400 words, 60 overlap) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Index | FAISS `IndexFlatIP` (cosine via normalisation) |
| Sparse Retrieval | TF-IDF + bigrams (scikit-learn), cosine similarity |
| LLM Generation | Ollama (llama3:8b, local) + extractive zero-hallucination fallback |
| Citations | Document name + page number validated against retrieved chunks |
| Abstention | Returns "not found" when similarity < threshold — no hallucination |
| Evaluation | Recall@1/3/5, MRR, Token F1, Citation Accuracy, Abstention Accuracy |
| UI | Streamlit multi-page, ChatGPT-style bubbles, sidebar nav |
| Tests | 32 pytest tests, no paid API required |

---

## 🧰 Technology Stack

| Component | Technology |
|---|---|
| UI Framework | Streamlit 1.35+ |
| PDF Extraction | PyMuPDF (fitz) |
| Embeddings | Sentence Transformers |
| Vector Search | FAISS (faiss-cpu) |
| Sparse Retrieval | scikit-learn TF-IDF |
| LLM (optional) | Ollama (local, free) |
| Data Processing | pandas, NumPy |
| Visualisation | Matplotlib |
| Testing | pytest |
| Config | python-dotenv, PyYAML |
| Validation | Pydantic |

---

## 📁 Project Structure

```
EduDocAI/
├── app.py                          # Streamlit multi-page application (3 pages)
├── requirements.txt
├── README.md
├── .env.example                    # Environment variable template
├── .gitignore
├── assets/
│   └── logo.png                    # EduDocAI brand logo
│
├── config/
│   └── settings.py                 # Centralised configuration dataclass
│
├── src/
│   ├── ingestion/
│   │   ├── document_loader.py      # PDF & TXT loading (PyMuPDF)
│   │   └── text_cleaner.py         # Unicode normalisation, whitespace
│   ├── preprocessing/
│   │   └── chunking.py             # Sliding-window sentence-aware chunker
│   ├── embeddings/
│   │   └── embedding_model.py      # Sentence Transformer singleton
│   ├── retrieval/
│   │   ├── dense_retriever.py      # FAISS vector retrieval
│   │   ├── tfidf_retriever.py      # TF-IDF sparse retrieval
│   │   ├── document_filter.py      # Source-level pre-filtering
│   │   └── reranker.py             # Hybrid dense+sparse reranker
│   ├── generation/
│   │   ├── answer_generator.py     # Grounded answer generation engine
│   │   ├── ollama_client.py        # Ollama HTTP client (cached availability)
│   │   └── grounding_validator.py  # Hallucination detection & abstention
│   ├── evaluation/
│   │   ├── evaluator.py            # End-to-end evaluation pipeline
│   │   └── metrics.py              # Recall@K, MRR, Token F1, etc.
│   └── utils/
│       ├── validators.py           # File type/size validation
│       ├── text_analytics.py       # Keyword extraction, domain classification
│       └── logger.py               # Structured logging
│
├── scripts/
│   ├── prepare_dataset.py          # Dataset audit & manifest
│   ├── build_index.py              # Build and save FAISS + TF-IDF indexes
│   └── run_evaluation.py           # CLI evaluation pipeline
│
├── data/
│   ├── sample_documents/           # 3 benchmark PDFs (TechCorp, AI Ethics, Whitepaper)
│   └── evaluation/
│       └── eval_dataset.json       # 22 annotated ground-truth test questions
│
├── results/
│   └── metrics/
│       └── evaluation_summary.json # Pre-computed evaluation results
│
├── tests/
│   ├── test_chunking.py
│   ├── test_retrieval.py
│   ├── test_generation.py
│   ├── test_ingestion.py
│   ├── test_filtering.py
│   └── test_evaluation.py
│
├── notebooks/
│   └── exploratory_analysis.ipynb  # EDA notebook
│
└── reports/
    ├── technical_report.md
    └── demo_script.md
```

---

## 🚀 Installation & Setup

### Prerequisites

- Python 3.10 or 3.11
- Windows / macOS / Linux
- (Optional) [Ollama](https://ollama.ai) for local LLM generation

### 1. Clone the repository

```bash
git clone https://github.com/guptaKartikey/EduDocAI.git
cd EduDocAI
```

### 2. Create a virtual environment

```powershell
# Windows (PowerShell)
python -m venv venv
venv\Scripts\activate
```

```bash
# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** The first run downloads the `all-MiniLM-L6-v2` embedding model (~90 MB). This is cached automatically at `~/.cache/huggingface/`.

### 4. Configure environment variables (optional)

```bash
copy .env.example .env   # Windows
cp .env.example .env     # macOS/Linux
# Edit .env to configure Ollama endpoint or OpenAI-compatible API
```

---

## 📂 Dataset

The project ships with **three sample benchmark PDFs** in `data/sample_documents/`:

| Document | Description | Licence |
|---|---|---|
| `TechCorp_Employee_Handbook_2025.pdf` | Workplace policies, benefits, remote work | MIT (project-generated) |
| `Global_AI_Ethics_Guidelines_2025.pdf` | AI governance, fairness, accountability | MIT (project-generated) |
| `EduDocAI_Architecture_Whitepaper.pdf` | System design and RAG architecture | MIT (project-generated) |

The evaluation dataset (`data/evaluation/eval_dataset.json`) contains **22 annotated ground-truth Q&A pairs** covering:
- Direct Factual (9 questions)
- Numeric / Quantitative (4 questions)
- Cross-Document (5 questions)
- Abstention / Out-of-scope (4 questions)

---

## ▶️ Running the Application

```bash
streamlit run app.py
```

App opens at `http://localhost:8501`.

**Pages:**
1. 🏠 **Home** — Upload your own PDFs **or** click "Load 3 Benchmark Sample Documents" for instant demo
2. 💬 **Chat with Documents** — ChatGPT-style Q&A with citations and grounding score
3. 📊 **Evaluation** — Full benchmark metrics dashboard with error analysis

---

## 🧪 Running Evaluation

### From the UI
Go to **Evaluation → Run Evaluation Now**.

### From the command line

```bash
# Prepare and audit the dataset
python scripts/prepare_dataset.py

# Build indexes (optional, app builds them automatically)
python scripts/build_index.py --chunk-size 400 --chunk-overlap 60

# Run the full evaluation pipeline
python scripts/run_evaluation.py --top-k 3
```

Results are saved to `results/metrics/evaluation_summary.json`.

---

## ✅ Running Tests

```bash
# Run all 32 tests
python -m pytest tests/ -v

# Run specific module
python -m pytest tests/test_retrieval.py -v
```

All tests run **without paid API access** — LLM calls are mocked.

**Expected:** `32 passed` in ~17 seconds

---

## 🤖 Optional: Local LLM with Ollama

To enable full generative answers (instead of extractive fallback):

```bash
# Install Ollama from https://ollama.ai
ollama pull llama3:8b

# Start server (runs in background)
ollama serve
```

The app auto-detects Ollama on port 11434 and switches to generative mode.

---

## 📊 Evaluation Results

| Metric | TF-IDF Baseline | Dense RAG | Advantage |
|---|---|---|---|
| Recall@1 | 86.4% | 86.4% | Parity |
| Recall@3 | **100%** | **100%** | Parity |
| Recall@5 | **100%** | **100%** | Parity |
| MRR | 0.9242 | **0.9318** | Dense +0.82% |
| Token F1 | 48.96% | **50.26%** | Dense +2.60% |
| Citation Accuracy | 86.36% | **90.91%** | Dense +4.55% |
| Abstention Accuracy | 86.36% | **90.91%** | Dense +4.55% |
| Avg Groundedness | 94%+ | 94%+ | Both high |
| Retrieval Latency | ~2.9 ms | ~30.8 ms | TF-IDF faster |

> Dense RAG outperforms TF-IDF on all semantic quality metrics. TF-IDF is ~10x faster for retrieval.

---

## 🛠️ Troubleshooting

| Issue | Solution |
|---|---|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| Model download slow | First run only — cached at `~/.cache/huggingface/` |
| PDF returns no text | PDF may be scanned — OCR not yet supported |
| Ollama not detected | Ensure `ollama serve` is running on port 11434 |
| FAISS import error | Run `pip install faiss-cpu` |
| Windows path errors | Use PowerShell, not CMD |
| Port 8501 in use | Run `streamlit run app.py --server.port 8502` |

---

## 🤝 AI Assistance Disclosure

This project was built with AI coding assistance (Google Gemini / Antigravity). All code has been reviewed, understood, and can be defended in a technical interview. The architecture decisions, metric implementations, chunking strategy, grounding validation logic, and evaluation design are original engineering choices made for this assessment.

---

## 👤 Author

**Kartikey Gupta**  
B.Tech Computer Science Engineering  
📧 k78048101@gmail.com  
🔗 [LinkedIn](https://linkedin.com/in/kartikey-gupta-988206372)  
🐙 [GitHub](https://github.com/guptaKartikey/EduDocAI)

---

## 📄 Licence

MIT © 2026 — Kartikey Gupta
