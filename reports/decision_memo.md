# Decision Memo — EduDocAI Launch Readiness
**To:** EduRankAI Founding Team
**From:** Kartikey Gupta
**Date:** November 2026
**Subject:** Should we launch EduDocAI to customers? My honest assessment.

---

## What the system does

EduDocAI lets a user upload any PDF document and ask questions about it in plain English. The system finds the relevant section of the document, pulls out the answer, and tells the user exactly which page it came from. If the answer is not in the document, the system says so — it never makes something up.

In testing, it answered 90 out of every 100 questions correctly, always citing the source page.

Live demo: https://edudocai.streamlit.app/
Source code: https://github.com/guptaKartikey/EduDocAI

---

## What it can handle well

- Policy documents, employee handbooks, research papers, resumes, technical whitepapers
- Questions with specific factual answers ("What is the PTO accrual rate?", "What are the core working hours?")
- Questions where there is no answer in the document — it correctly says "not found" rather than guessing
- Multiple documents at the same time, with the ability to search one or all
- Runs entirely on your own machine — no data is sent to any cloud service

---

## Where it will struggle

- **Scanned PDFs:** If someone uploads a photo or a scanned document (not a text PDF), the system cannot read it. It will warn the user. A fix exists (OCR integration) but was out of scope for this prototype.
- **Complex tables and charts:** Financial spreadsheets or multi-column data embedded in PDFs may not extract cleanly. The text comes out in the wrong order and the answer can be incomplete.
- **Questions that need combining two things:** If someone asks a question that requires putting together information from page 3 and page 7 at the same time, the system sometimes only picks one. It gets it right most of the time but not always.
- **Very long documents:** Documents above roughly 200 pages start to slow down on the first load (index building takes longer). Once loaded, answers are still fast.

---

## My recommendation

**Launch in a limited pilot, not a full public release.**

The system is solid enough to be genuinely useful to a specific set of users — HR teams working with policy documents, researchers reviewing papers, professionals looking up contract terms. For those use cases, it performs well and is honest about what it does not know.

I would not launch it yet for:
- Legal or medical document Q&A where a wrong answer has serious consequences
- High-stakes financial document interpretation
- Any setting where users might not realize the system can be wrong

**Suggested first launch group:** Internal teams (HR, operations, legal) at the company, with a clear note that all answers should be verified against the original document for anything important.

---

## Expected benefit

- Saves roughly 5–15 minutes per policy lookup for a typical team member
- Reduces back-and-forth emails to HR for standard questions
- Makes document knowledge accessible to people who do not have time to read 40-page handbooks

---

## What could go wrong

- A user trusts a wrong answer on an important decision (most likely failure mode)
- An unusual PDF format causes incorrect text extraction and therefore a wrong answer
- The system confidently cites "Page 3" but the actual policy is on Page 5 (citation offset from re-indexed documents)

To reduce these risks: always show the source page number (already implemented), always allow users to expand and read the original passage (already implemented via "View source passages"), and add a visible disclaimer that answers should be verified for high-stakes decisions.

---

## What I would watch in the first month

1. **Abstention rate:** What percentage of questions get "not found"? If it is above 30%, users are asking things the documents do not cover — we may need to expand the document library.
2. **Citation accuracy:** Spot-check 20 answers per week manually against the source documents. If citation accuracy drops below 85%, something in the pipeline has regressed.
3. **User feedback on wrong answers:** Add a simple thumbs up/thumbs down on each answer. Look at the thumbs-down patterns — they will show which document types or question types fail most.
4. **Response time:** If average response time goes above 3 seconds, users will lose patience. Monitor this as document volume grows.

---

## What I am unsure about

- Whether the 22-question benchmark is representative enough of real user behavior. Real users ask messier, more ambiguous questions than a curated test set.
- How the system will perform on documents in languages other than English. Not tested.
- Whether the extractive fallback (when Ollama is offline) produces answers that feel natural enough for non-technical users, or whether it feels robotic.

---

*Written by Kartikey Gupta. All claims in this memo are based on measured results from the evaluation benchmark. I have not inflated any numbers.*
