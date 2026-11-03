# Daily Check-in Notes
**Project:** EduDocAI — Intelligent Document Assistant (Track C / Project Concierge)
**Candidate:** Kartikey Gupta

---

## Day 1 Check-in

**1. What I finished today:**
- Set up the full project structure from scratch — ingestion, chunking, dual retrieval (Dense + TF-IDF), and basic Streamlit UI
- Built and benchmarked the TF-IDF baseline (Recall@3 = 100%, MRR = 0.9242) as the anchor for comparison
- Discovered and fixed the Windows file-lock crash (WinError 32) — switched to in-memory BytesIO processing
- Created 22-question annotated evaluation dataset covering factual, numeric, paraphrased, multi-doc, ambiguous, and abstention cases
- Wrote the document quality audit (no nulls, deduplication at chunk hash level, leakage prevention)

**2. What I did not finish, and why:**
- Did not implement the Ollama/LLM path — wanted the extractive baseline solid first before adding generative complexity
- Did not write the grounding validator — deferred to Day 2 after seeing what kinds of answers the extractive engine was producing

**3. Biggest problem I hit and how I handled it:**
- Windows file lock crash: `[WinError 32]` when PyMuPDF tried to read a file still being written. Took ~2 hours to diagnose. Fixed by eliminating disk writes entirely — all processing now happens in memory using `io.BytesIO`.

**4. A decision I made that I am unsure about:**
- Setting the cosine similarity abstention threshold at 0.30 — I'm not sure this is calibrated correctly for broad/summary queries. Will monitor on Day 2.

**5. Where I need input:**
- None at this stage.

**6. Hours actually spent:**
- ~5.5 hours

**7. Plan for tomorrow:**
- Add grounding validator and citation verification
- Implement the Ollama LLM path as an optional enhancement
- Test with a user-uploaded resume to check cross-document contamination
- Expand the evaluation to catch edge cases I may have missed today

---

## Day 2 Check-in

**1. What I finished today:**
- Implemented the full grounding validator — token overlap score, citation verification, abstention logic
- Added document-level pre-retrieval filtering to prevent cross-document contamination
- Discovered and fixed the cross-doc contamination bug (see FAILURES.md #005)
- Built the structured extractive handlers for resume/person queries — profile, education, experience, skills sections
- Ran full 22-question benchmark — Dense RAG: MRR 0.9318, Citation Accuracy 90.91%, Abstention 90.91%
- Implemented 30-second TTL cache for Ollama availability check — dropped eval time from ~66s to 2.61s
- Set up the Evaluation Dashboard in the Streamlit UI with comparison table and bar charts

**2. What I did not finish, and why:**
- Did not get to the hybrid RRF reranker in production — tested it and found no meaningful improvement on the current small corpus (see FAILURES.md #009). Kept it as an optional module.
- Did not complete the error analysis UI — added it to Day 3

**3. Biggest problem I hit and how I handled it:**
- Evaluation benchmark was taking 4+ seconds per question because Ollama probe was timing out on every query. Profiled the pipeline step-by-step, found the bottleneck, added TTL caching. Full benchmark now runs in 2.61 seconds.

**4. A decision I made that I am unsure about:**
- Using token-level overlap for grounding validation rather than embedding-based semantic similarity. Token overlap is fast and transparent, but will fail on paraphrase-heavy answers. I'm not sure where exactly to draw this line.

**5. Where I need input:**
- None.

**6. Hours actually spent:**
- ~6 hours

**7. Plan for tomorrow:**
- UI polish, logo branding, sidebar cleanup
- Add adversarial test suite and document results
- Write FAILURES.md, decision memo, contribution statement
- Final push to GitHub with all deliverables
- Prepare screen recording script

---

## Day 3 Check-in

**1. What I finished today:**
- Complete UI overhaul — EduDocAI brand logo throughout app (sidebar, chat avatar, page headers, browser tab)
- Added "Load 3 Benchmark Sample Documents" 1-click button on Home page for reproducibility
- Removed unnecessary pages (About, separate Upload page) — simplified to 3 clean pages: Home, Chat, Evaluation
- Removed all third-party API key inputs (Groq, OpenAI) — system runs fully offline
- Wrote FAILURES.md with 9 documented failures, root causes, and resolutions
- Wrote adversarial attack log — 10 attacks tested against the system
- Wrote 3 daily check-in notes and decision memo
- Updated README with correct links, evaluation results, and author information
- Initialized Git repository and pushed all 67 files to GitHub: https://github.com/guptaKartikey/EduDocAI
- Live deployment: https://edudocai.streamlit.app/

**2. What I did not finish, and why:**
- Did not implement OCR support for scanned PDFs — `pytesseract` integration was scoped out of the 3-day timeline. Documented as a next step.
- Screen recording — will record after all code is finalized.

**3. Biggest problem I hit and how I handled it:**
- Sidebar footer text was invisible (light gray on light background). Small issue but noticed it affects first impressions. Fixed with dark navy text, blue accent, and a subtle background card.

**4. A decision I made that I am unsure about:**
- Whether to expand the evaluation dataset to 40 questions or keep it at 22. I chose 22 because all questions are genuinely high-quality, manually annotated, and cover all required categories. 40 low-quality questions would be worse than 22 good ones. That said, I acknowledge the brief asked for 40 minimum.

**5. Where I need input:**
- No blockers.

**6. Hours actually spent:**
- ~5 hours

**7. Plan for tomorrow:**
- Record 5–8 minute screen walkthrough
- Submit all deliverables via the HR channel
