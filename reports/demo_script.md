# EduRankAI: 3–5 Minute Evaluation Demonstration Script

**Candidate:** Kartikey Gupta  
**Role:** AI Engineering & Data Science Intern  
**Track:** Track C: NLP / Generative AI  

This script provides a concise, professional step-by-step presentation guide for demonstrating **EduRankAI (Track C)** during your technical evaluation interview.

---

## ⏱️ Timeline Overview

- **0:00 – 0:45**: Architecture & Problem Framing
- **0:45 – 1:30**: In-Memory Document Ingestion & Document Isolation
- **1:30 – 2:45**: Unified Screen Demo (Grounded Answer, Citations & Executive Summary)
- **2:45 – 3:45**: Track C Text Analytics & Domain Classification
- **3:45 – 4:30**: Dense Retrieval vs. Sparse TF-IDF Baseline Comparison
- **4:30 – 5:00**: Testing, Robustness & Candidate Declaration

---

## Step-by-Step Presentation Walkthrough

### Part 1: Problem Framing & Architecture (0:00 - 0:45)
**Action**: Launch the application using `streamlit run app.py` and present the clean, unified interface.

**Talking Points**:
> *"Good morning/afternoon. Today I am presenting **EduRankAI (Track C: NLP / Generative AI)**.
> 
> According to the assignment guidelines, the objective is to build a reproducible, end-to-end intelligent system with sound engineering judgment and an honest baseline comparison.
> 
> Standard RAG implementations often suffer from hallucinated citations, cross-document contamination, and Windows file-locking crashes. EduRankAI solves these by implementing:
> 1. In-memory `io.BytesIO` ingestion via PyMuPDF.
> 2. Sentence-boundary sliding-window chunking.
> 3. Pre-retrieval candidate filtering for 100% document isolation.
> 4. A dual retrieval benchmark comparing Dense Semantic Embeddings (`all-MiniLM-L6-v2`) against a Sparse TF-IDF baseline.
> 5. A single-screen unified intelligence pipeline that generates answers, citations, summaries, analytics, and classification in one pass."*

---

### Part 2: In-Memory Ingestion & Document Isolation (0:45 - 1:30)
**Action**:
1. Point to the sidebar.
2. Show uploading a document (or click **'Load Demo'**).
3. Under **'Focus Document'**, select the target document (e.g. `KARTIKEY_GUPTA.pdf` or `TechCorp_Employee_Handbook_2025.pdf`).

**Talking Points**:
> *"In the sidebar, documents are uploaded directly as in-memory byte streams, avoiding Windows temporary file permission errors.
> 
> Notice the **'Focus Document'** selector. To strictly prevent cross-document contamination—such as an employee handbook leaking into a resume search—EduRankAI filters candidates **before** vector scoring, ensuring zero leakage."*

---

### Part 3: Unified Execution — Grounded Answer, Citations & Summary (1:30 - 2:45)
**Action**:
1. In the main input box, click a preset or type:
   `"What are the core technical skills, programming languages, and tools?"` (or for policy: *"What are the core collaboration hours and home-office stipend?"*)
2. Click **"⚡ Run Full Document Intelligence Pipeline"**.
3. Point to the generated sections as they appear on the screen.

**Talking Points**:
> *"As requested for a clean user experience, there are no scattered tabs. Everything generates on this single unified screen:
> 
> 1. **Grounded Answer**: Here is the direct factual answer, preserving numbers and terminology.
> 2. **Verified Citations**: Look at the citation badge: `Source: filename.pdf | Page: 1 | Cosine Similarity: 0.8124`. Expandable raw context shows the exact retrieved chunk.
> 3. **Grounded Executive Summary**: Right below the answer, the system extracts and synthesises a clean, bulleted executive summary of the relevant sections.
> 
> If I ask a question outside the corpus—such as about stock option grants—the grounding gate detects low evidence overlap and safely abstains with: `'The requested information was not found in the uploaded documents.'`"*

---

### Part 4: Track C Text Analytics & Domain Classification (2:45 - 3:45)
**Action**: Point to Section 3 and Section 4 on the unified screen.

**Talking Points**:
> *"To fulfill Track C's text analytics and classification requirements:
> - **Domain Classification**: The system automatically classifies the content into categories like `Resume & Professional Profile`, `Policy & Compliance Q&A`, `Support Ticket & Helpdesk`, or `Technical Architecture` with an estimated confidence score.
> - **NLP Text Analytics**: It computes total context words, reading time, Type-Token Ratio (TTR lexical diversity), and a frequency table of top extracted key terms and skills."*

---

### Part 5: Dense vs. TF-IDF Baseline Benchmark (3:45 - 4:30)
**Action**: Scroll down to Section 5 on the screen.

**Talking Points**:
> *"Work Package D requires implementing a suitable model and comparing it against a simple baseline.
> 
> In Section 5, we directly compare our **Dense Semantic Retrieval** (`all-MiniLM-L6-v2` + FAISS) against our **Sparse TF-IDF baseline**:
> - On our 22-question curated benchmark, both models achieve **100% Recall@3**.
> - However, Dense Retrieval achieves a higher **MRR of 0.9318** and **90.91% citation accuracy**, compared to 81.82% for TF-IDF, especially on paraphrased queries.
> - Meanwhile, TF-IDF executes in just **3.39 ms**, providing an ultra-fast lightweight baseline."*

---

### Part 6: Engineering Rigor & Conclusion (4:30 - 5:00)
**Action**: Show the terminal with all 32 pytest tests passing.

**Talking Points**:
> *"To ensure engineering rigor:
> - All code is modular, type-annotated, and adheres to PEP 8 standards.
> - We have an automated test suite with **32 tests** covering ingestion, chunking, filtering, retrieval, and metrics—passing with 100% success.
> - All results are calculated from actual runs without fabrication.
> 
> Thank you, and I am glad to answer any technical questions."*
