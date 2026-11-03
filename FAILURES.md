# FAILURES.md — Honest Engineering Log
**Project:** EduDocAI (Track C: NLP / Generative AI)
**Author:** Kartikey Gupta
**Period:** Day 1 → Day 3

---

> This file documents every time something broke, surprised me, or forced me to change direction.
> Written honestly — not to look good, but because understanding failure is how real engineering works.

---

## FAILURE 001 — Windows File-Lock Crash on PDF Upload

**When:** Day 1, early setup
**What happened:** The first version of the document loader saved uploaded PDFs to a temporary file on disk before reading them with PyMuPDF. On Windows, the OS keeps an exclusive lock on files that are being written. When PyMuPDF tried to open the same file immediately after writing, it threw `[WinError 32]: The process cannot access the file because it is being used by another process.`

**What I believed:** I thought writing to a temp file was the standard approach since most tutorials do it that way.

**What I tried:**
- Adding `time.sleep(0.1)` between write and read — didn't fix it
- Using `NamedTemporaryFile(delete=False)` — helped sometimes but was inconsistent
- Finally: Switched to fully in-memory processing using `io.BytesIO`. The uploaded file bytes are passed directly to `fitz.open(stream=..., filetype="pdf")` without ever touching disk.

**What I learned:** Windows file locking is stricter than Linux/macOS. Any production code targeting Windows must prefer stream-based processing over temp-file approaches for binary data.

**Status:** Fixed. In-memory BytesIO approach is now the default.

---

## FAILURE 002 — Cosine Similarity Threshold Too High — Good Answers Rejected

**When:** Day 1, testing retrieval
**What happened:** I set the initial abstention threshold at `similarity_score < 0.30`. When testing with a question like `"what is the pdf about"`, the system returned the abstention message even though the document was fully loaded and relevant. Cosine similarity for general/broad queries was often 0.08–0.18, which fell below the threshold.

**What I believed:** A score below 0.30 meant the retrieval was genuinely uncertain, so abstaining was correct.

**What I tried:**
- Checked the raw retrieved chunks — they were actually relevant, just broad
- Plotted the distribution of cosine scores across all 22 eval questions
- Found that factual, specific queries scored 0.65–0.85, but broad/summary queries scored 0.08–0.20 — both valid, just different in nature

**What I learned:** Cosine similarity threshold should not be the only abstention signal. A better approach is to use it together with token overlap (how many query words appear in the retrieved chunk). I reduced the hard threshold to 0.15 and added a secondary grounding score check.

**Status:** Fixed. Dual-check: cosine threshold + grounding validator token overlap.

---

## FAILURE 003 — `NameError: name 're' is not defined` in Production

**When:** Day 2, running the chat page
**What happened:** After a refactor where I reorganized the import order at the top of `app.py`, I accidentally moved the `import re` inside a function scope. The `_render_message_html()` function called `re.sub()` but `re` was not in scope at runtime. This caused the entire chat page to crash with a `NameError`.

**What I believed:** I assumed Python's top-level imports were always globally available.

**What I tried:**
- Traced the traceback to line 249 in `app.py`
- Added `import re` and `import html` explicitly at the top of the file
- Added a null/empty content guard inside `_render_message_html()`

**What I learned:** When refactoring, always run a full page load test after moving imports — Python only raises `NameError` at call time, not at import time. Added this to my mental checklist.

**Status:** Fixed. Both `re` and `html` are explicitly imported at file top.

---

## FAILURE 004 — Evaluation Taking 4+ Seconds Per Query (Ollama Probe Latency)

**When:** Day 2, running the 22-question evaluation benchmark
**What happened:** Each question in the benchmark was taking 3.5–4.0 seconds to process even though Ollama was not running (offline mode). The bottleneck was that `OllamaClient.is_available()` was making a fresh HTTP connection attempt to `localhost:11434` every single time — which timed out with a 3-second socket timeout per query.

**What I believed:** Checking availability once per query seemed safe and defensive.

**What I tried:**
- Profiled using `time.perf_counter()` around each pipeline step
- Isolated the bottleneck to the Ollama availability probe
- Implemented a 30-second TTL cache using `functools.lru_cache` equivalent with a timestamp check

**What I learned:** Any I/O check (network, file, socket) that runs in a tight evaluation loop must be cached. Even a 3-second timeout × 22 queries = 66 seconds of pure waiting. After the fix, the full benchmark runs in 2.61 seconds.

**Status:** Fixed. Ollama availability is now cached for 30 seconds — sub-millisecond on repeated calls.

---

## FAILURE 005 — Cross-Document Answer Contamination

**When:** Day 1, testing with two PDFs simultaneously
**What happened:** I uploaded the TechCorp Employee Handbook and the AI Ethics Guidelines together. When I asked `"What are the data privacy requirements?"`, the system retrieved a chunk from the Ethics document even though the user had selected the Handbook in the filter. The answer mixed policies from two unrelated documents.

**What I believed:** The vector search would naturally rank the most similar chunk regardless of document — which is correct behavior for global search, but wrong for document-specific queries.

**What I tried:**
- Added a pre-retrieval document filter that masks out all chunk indices not belonging to the selected document(s) before cosine similarity is computed
- This ensures the similarity matrix is only computed over the valid candidate set

**What I learned:** In multi-document RAG systems, pre-retrieval filtering is essential when users expect document-scoped answers. Post-retrieval reranking alone is not sufficient because irrelevant documents can still score higher than relevant ones on generic queries.

**Status:** Fixed. Document-level pre-filtering is applied before any embedding computation.

---

## FAILURE 006 — "About Kartikey" Returned Raw Email/Phone Dump

**When:** Day 2, testing with resume PDF
**What happened:** When a user asked `"about kartikey gupta"`, the system returned a raw string of `"k78048101@gmail.com LinkedIn: linkedin.com/... KARTIKEY GUPTA INTERNSHIP EXPERIENCE..."` — essentially the raw first chunk of the resume without any structure or formatting.

**What I believed:** The extractive fallback would naturally pick the most informative sentences.

**What I tried:**
- Analyzed why: the first chunk of a resume contains contact info at the top, which was being returned as-is
- Added `_clean_sentence()` to strip emails, URLs, phone numbers, and all-caps section headers (like "EDUCATION", "SKILLS") that appear as standalone lines
- Added a `_is_noise()` filter for fragments that are purely location strings (like "Jamshedpur, Jharkhand, India") with no informational content
- Added a dedicated `_answer_person_query()` handler that categorizes sentences by section (profile, education, experience) and builds a structured bullet-point response

**What I learned:** Resume/CV documents have very different structure from prose documents. A general extractive engine needs document-type-aware post-processing to produce readable output.

**Status:** Partially fixed. Profile queries now return structured bullet points with section headers. Some edge cases remain with mixed formatting PDFs.

---

## FAILURE 007 — Broad Queries Like "What is the PDF About?" Failed Abstention Check

**When:** Day 2, user testing
**What happened:** When a user asked `"what is the pdf about"`, the grounding validator would sometimes reject the answer even though the extractive summary was pulling real content from the document. The validator was checking if answer tokens appeared in the context, but broad summary answers use transitional phrases ("this document covers", "the key topics include") that don't appear verbatim in the source text.

**What I believed:** Token overlap is the right measure of grounding for all query types.

**What I tried:**
- Added special handling: if `_is_summary_query(query)` returns True, the grounding threshold is relaxed to 0.25 (from 0.30) since summary answers naturally paraphrase
- Added summary-query detection (`"what is this about"`, `"summarize"`, `"overview"`, etc.) as a separate intent path

**What I learned:** Grounding validation thresholds should be query-type-aware. Factual queries need strict token matching; summary queries need semantic overlap, not lexical overlap.

**Status:** Partially fixed. Summary path has relaxed threshold. Full semantic grounding would require embedding-based rather than token-based validation.

---

## FAILURE 008 — Sidebar Footer Text Invisible on Dark Background

**When:** Day 3, UI polish
**What happened:** The sidebar footer showed `"EduDocAI v2.0"` in `color: #D1D5DB` (light gray), which was nearly invisible against the light blue sidebar background (`#F0F4FF`). Multiple users in testing could not read the footer text at all.

**What I tried:**
- Changed text color to `#1E3A5F` (dark navy) for the "Developed By" line
- Changed "Powered by RAG + LLM" to `#2563EB` (brand blue) with `font-weight: 700`
- Added a light blue background card (`#EFF6FF`) with a blue top border to the footer block

**Status:** Fixed.

---

## FAILURE 009 — Abandoned Approach: Hybrid RRF Reranker in Default Pipeline

**When:** Day 2, architecture decision
**What happened:** I implemented a Reciprocal Rank Fusion (RRF) hybrid reranker that combined Dense + TF-IDF scores. On paper this should improve performance. In practice, for the current corpus (3 small documents, < 50 chunks), RRF gave identical results to pure dense retrieval but added 8–12ms latency per query.

**What I decided:** Keep RRF as a module (`src/retrieval/reranker.py`) that can be enabled via config, but default to pure dense retrieval for the current corpus size. At scale (10,000+ chunks), RRF would provide meaningful diversity benefit.

**What I learned:** Ensemble methods only help when there is genuine diversity in what the two systems retrieve. On small, clean corpora, they add complexity without benefit.

**Status:** RRF implemented but not active by default. Toggle via `settings.USE_HYBRID_RERANKER`.

---

## Summary

| # | Type | Impact | Resolution |
|---|---|---|---|
| 001 | OS/Platform Bug | App crash on Windows | Fixed — in-memory BytesIO |
| 002 | Threshold Design | Good answers rejected | Fixed — dual-check |
| 003 | Import Error | Chat page crash | Fixed — explicit imports |
| 004 | Performance | 4s per query in eval | Fixed — 30s TTL cache |
| 005 | Retrieval Logic | Cross-doc contamination | Fixed — pre-retrieval filter |
| 006 | Output Quality | Raw data dump | Partially fixed — structured handler |
| 007 | Grounding Logic | Summary queries rejected | Partially fixed — relaxed threshold |
| 008 | UI/UX | Invisible footer text | Fixed — color + styling |
| 009 | Architecture | Premature complexity | Deprioritized — kept as optional module |
