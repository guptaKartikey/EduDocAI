"""Grounded answer generation engine with LLM (OpenAI/Groq/Ollama) and natural-language synthesis.

Architecture
────────────
1.  LLM path  : If an API key (OpenAI / Groq) or local Ollama is configured, send the
                retrieved chunks as context and get a fully-worded answer.
2.  Extractive path (offline)  : When no LLM is available the system falls back to a
                structured rule-based synthesiser that
                  * detects the query intent (person/profile, skills, projects, education,
                    summary, or factual)
                  * selects the most informative sentences from retrieved chunks
                  * strips contact noise (emails, phone numbers, LinkedIn URLs, section headers)
                  * frames a clean natural-language response with citations

Grounding Validator runs on both paths before the answer is returned, and will
substitute the abstention message if the groundedness score is too low.
"""

import re
from typing import List, Dict, Any, Optional

from dataclasses import dataclass

from config.settings import settings
from src.retrieval.dense_retriever import SearchResult
from src.generation.ollama_client import OllamaClient
from src.generation.llm_provider import (
    BaseLLMProvider,
    OpenAICompatibleProvider,
    OllamaProvider,
    get_active_llm_provider,
)
from src.generation.grounding_validator import GroundingValidator, ValidationReport
from src.utils.logger import logger


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GeneratedAnswer:
    """Structured response containing generated answer, citations, and grounding audit."""

    question: str
    answer: str
    citations: List[Dict[str, Any]]
    sources_used: List[str]
    is_abstained: bool
    groundedness_score: float
    generation_mode: str
    retrieved_chunks: List[SearchResult]
    validation_notes: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert answer structure to a serialisable dictionary."""
        return {
            "question": self.question,
            "answer": self.answer,
            "citations": self.citations,
            "sources_used": self.sources_used,
            "is_abstained": self.is_abstained,
            "groundedness_score": self.groundedness_score,
            "generation_mode": self.generation_mode,
            "validation_notes": self.validation_notes,
            "retrieved_chunk_count": len(self.retrieved_chunks),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Vocabulary helpers
# ─────────────────────────────────────────────────────────────────────────────

# Words that a user uses to refer to the document itself – not real content words
META_QUERY_WORDS: frozenset = frozenset(GroundingValidator.STOPWORDS) | {
    "pdf", "pdfs", "document", "documents", "doc", "docs", "file", "files",
    "page", "pages", "paper", "papers", "text", "uploaded", "tell", "give",
    "show", "explain", "describe", "detail", "details", "mention", "mentioned",
    "say", "says", "stated", "state", "find", "information", "info", "know",
    "please", "about", "all", "what", "is",
}

# Section-header labels commonly found in resumes / reports
_SECTION_HEADERS = frozenset({
    "profile", "internship experience", "education", "certifications",
    "technical skills", "projects", "experience", "summary", "objective",
    "contact", "references", "achievements", "awards",
})

# Regex patterns
_RE_EMAIL   = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_RE_URL     = re.compile(r"(?:https?://|www\.|linkedin\.com/)\S+", re.IGNORECASE)
_RE_PHONE   = re.compile(r"\+?\d[\d\s\-]{8,}\d")
# Only remove ALL-CAPS section headers when they appear as the sole content of a
# short phrase (≤ 4 words), not when they are embedded inside regular sentences.
_RE_HEADER  = re.compile(
    r"^(PROFILE|INTERNSHIP\s+EXPERIENCE|EDUCATION\s*/?\\s*CERTIFICATIONS?"
    r"|TECHNICAL\s+SKILLS|PROJECTS|EXPERIENCE|CERTIFICATIONS?|SUMMARY)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_RE_BULLETS = re.compile(r"^[•\-\*\s|:,]+|[•\-\*\s|:,]+$")
_RE_SPACES  = re.compile(r"\s+")


# ─────────────────────────────────────────────────────────────────────────────
# Main engine
# ─────────────────────────────────────────────────────────────────────────────

class AnswerGenerator:
    """Generates strictly grounded, well-framed natural language answers.

    When a live LLM is available (OpenAI, Groq, Ollama) it is used first.
    Otherwise the built-in Extractive Synthesiser produces a clean response
    without any risk of hallucination.
    """

    def __init__(
        self,
        ollama_client: Optional[OllamaClient] = None,
        llm_provider: Optional[BaseLLMProvider] = None,
        system_prompt: str = settings.SYSTEM_PROMPT,
    ):
        self.ollama_client  = ollama_client or OllamaClient()
        self.llm_provider   = llm_provider
        self.system_prompt  = system_prompt

    # ── Public API ────────────────────────────────────────────────────────────

    def format_context(self, retrieved_chunks: List[SearchResult]) -> str:
        """Formats retrieved chunks with clear source/page/ID tags for the LLM prompt."""
        blocks: List[str] = []
        for i, res in enumerate(retrieved_chunks, start=1):
            c = res.chunk
            header = f"--- [Document: {c.source} | Page: {c.page} | Chunk ID: {c.chunk_id}] ---"
            blocks.append(f"{header}\n{c.chunk_text.strip()}\n")
        return "\n".join(blocks)

    def generate_answer(
        self,
        query: str,
        retrieved_chunks: List[SearchResult],
        use_ollama: bool = True,
    ) -> GeneratedAnswer:
        """Generate a grounded answer for *query* from *retrieved_chunks*.

        Steps
        -----
        1. Abstain immediately if no chunks were retrieved.
        2. Try the configured LLM provider (OpenAI / Groq / Ollama).
        3. Fall back to the Extractive Synthesiser.
        4. Run GroundingValidator; override with abstention if score too low.
        """
        if not retrieved_chunks:
            logger.info("AnswerGenerator: No chunks retrieved – abstaining.")
            return self._abstain(query, note="No relevant chunks found in index for query.")

        citations    = self._build_citations(retrieved_chunks)
        sources_used = sorted({r.chunk.source for r in retrieved_chunks})

        is_summary = self._is_summary_query(query)
        answer_text: Optional[str] = None
        generation_mode = "extractive_framed"

        # ── Path 1: Live LLM ─────────────────────────────────────────────────
        active_provider = self.llm_provider or get_active_llm_provider()
        if active_provider and active_provider.is_available():
            answer_text, generation_mode = self._llm_generate(
                query, retrieved_chunks, active_provider, is_summary
            )

        # ── Path 2: Extractive Synthesiser ───────────────────────────────────
        if answer_text is None:
            if is_summary:
                answer_text    = self._extractive_summary(retrieved_chunks)
                generation_mode = "extractive_summary"
            else:
                answer_text    = self._extractive_fallback(query, retrieved_chunks)
                generation_mode = "extractive_framed"
            logger.info(f"Answer generated via {generation_mode}.")

        # ── Grounding validation ─────────────────────────────────────────────
        report = GroundingValidator.validate_answer(
            query=query,
            answer=answer_text,
            retrieved_chunks=retrieved_chunks,
        )

        return GeneratedAnswer(
            question=query,
            answer=report.final_answer,
            citations=citations if not report.is_abstained else [],
            sources_used=sources_used if not report.is_abstained else [],
            is_abstained=report.is_abstained,
            groundedness_score=round(report.groundedness_score, 4),
            generation_mode=generation_mode,
            retrieved_chunks=retrieved_chunks,
            validation_notes=report.notes,
        )

    # ── LLM generation helpers ────────────────────────────────────────────────

    def _llm_generate(
        self,
        query: str,
        retrieved_chunks: List[SearchResult],
        provider: BaseLLMProvider,
        is_summary: bool,
    ) -> tuple[Optional[str], str]:
        """Call the LLM provider and return (answer_text, generation_mode)."""
        context = self.format_context(retrieved_chunks)
        if is_summary:
            user_prompt = (
                f"Retrieved Document Context:\n{context}\n\n"
                "Task: Provide a concise, well-structured executive summary of the document. "
                "Include key highlights, background, and important details. "
                "Cite source and page numbers for factual sections using [Source: filename, Page: X]."
            )
        else:
            user_prompt = (
                f"Retrieved Document Context:\n{context}\n\n"
                f"Question: {query}\n\n"
                "Answer the question clearly in natural, complete, well-framed sentences "
                "using ONLY the document context above. "
                "Do not invent facts or use outside knowledge. "
                "Cite the source document and page number for each claim using [Source: filename, Page: X]."
            )
        response = provider.generate(prompt=user_prompt, system_prompt=self.system_prompt)
        if response:
            mode = f"llm_{provider.name}"
            logger.info(f"Answer generated via {mode}.")
            return response, mode
        return None, "extractive_framed"

    # ── Citation builder ──────────────────────────────────────────────────────

    @staticmethod
    def _build_citations(retrieved_chunks: List[SearchResult]) -> List[Dict[str, Any]]:
        return [
            {
                "source":   r.chunk.source,
                "page":     r.chunk.page,
                "chunk_id": r.chunk.chunk_id,
                "score":    round(float(r.score), 4),
                "rank":     r.rank,
            }
            for r in retrieved_chunks
        ]

    # ── Query intent detection ────────────────────────────────────────────────

    @staticmethod
    def _is_summary_query(query: str) -> bool:
        """Return True if the query is an overview / summarisation / document-level request."""
        q = query.strip().lower()
        triggers = [
            "summarize", "summary", "summarise", "overview", "brief overview",
            "give a summary", "key points", "highlights", "profile summary",
            "what is the pdf", "what is this pdf", "what is the document",
            "what is this document", "what is the file", "what is this file",
            "all about", "about the pdf", "about this pdf", "about the document",
            "about this document", "explain the pdf", "explain this pdf",
            "what does this document say", "what does this pdf say",
            "what is it about", "what's this about", "what is this about",
            "main objective", "main idea", "brief me", "executive summary",
            "tell me about this document", "tell me about this pdf",
        ]
        return any(kw in q for kw in triggers)

    @staticmethod
    def _detect_intent(query: str) -> str:
        """Classify query into a broad intent category for the extractive engine.

        Returns one of: 'person', 'skills', 'projects', 'education',
                        'experience', 'factual', 'summary'
        """
        q = query.strip().lower()
        if any(k in q for k in ["who is", "who are", "about", "profile", "background",
                                  "candidate", "applicant", "tell me about him",
                                  "tell me about her", "tell me about them"]):
            return "person"
        if any(k in q for k in ["skill", "skills", "technology", "technologies",
                                  "tech stack", "programming", "tools", "languages",
                                  "frameworks", "libraries", "known", "expertise"]):
            return "skills"
        if any(k in q for k in ["project", "projects", "built", "developed",
                                  "created", "portfolio", "work done"]):
            return "projects"
        if any(k in q for k in ["education", "degree", "university", "college",
                                  "b.tech", "btech", "qualification", "course",
                                  "study", "studied", "cgpa", "grade"]):
            return "education"
        if any(k in q for k in ["experience", "internship", "job", "worked",
                                  "company", "organisation", "organization",
                                  "employer", "intern", "role", "position"]):
            return "experience"
        return "factual"

    # ── Text cleaning helpers ─────────────────────────────────────────────────

    @staticmethod
    def _is_noise(text: str) -> bool:
        """Return True if the sentence is purely contact info, URL, or a lone section header."""
        t = text.strip()
        tl = t.lower()
        # Contains email or URL → always noise
        if _RE_EMAIL.search(t) or _RE_URL.search(t):
            return True
        # Pure phone number line
        if _RE_PHONE.fullmatch(t.strip()):
            return True
        # Lone header: ≤ 3 words AND the whole string is just a section label
        STANDALONE_HEADERS = {
            "profile", "internship experience", "education", "certifications",
            "technical skills", "projects", "experience", "summary", "objective",
            "contact", "references", "achievements", "awards",
            "education / certifications", "education/certifications",
        }
        words = tl.split()
        if len(words) <= 3 and tl.strip() in STANDALONE_HEADERS:
            return True
        # Junk location-only fragments like "Jamshedpur, Jharkhand, India"
        # (city, state, country – no meaningful content)
        if re.match(r"^[A-Za-z\s,]+,\s+[A-Za-z\s,]+,\s+[A-Za-z\s]+$", t) and len(words) <= 5:
            return True
        return False

    @staticmethod
    def _clean_sentence(text: str) -> str:
        """Strip emails, URLs, phone numbers, standalone section-header labels, and leading bullets."""
        s = text.strip()
        s = _RE_EMAIL.sub("", s)
        s = _RE_URL.sub("", s)
        s = _RE_PHONE.sub("", s)
        # Only strip a section header if it is the ONLY content of the string
        # (prevents stripping "experience" from "hands-on experience in...")
        words = s.strip().split()
        if len(words) <= 4 and _RE_HEADER.match(s.strip()):
            return ""
        s = _RE_BULLETS.sub("", s)
        s = _RE_SPACES.sub(" ", s).strip()
        return s

    def _split_sentences(self, text: str) -> List[str]:
        """Split raw chunk text into clean, informative sentences / bullet items.

        Strategy
        --------
        1. Split on newlines first (resume-style layouts: one idea per line).
        2. Then split any remaining long fragments on sentence-boundary punctuation.
        3. Strip contact noise, section headers, emails, URLs, phone numbers.
        4. Keep only items ≥ 12 characters with ≥ 3 alphabetic words.
        """
        # Step 1: newline split
        newline_parts = re.split(r"\n+", text)

        # Step 2: further split on sentence-ending punctuation
        raw_parts: List[str] = []
        for part in newline_parts:
            sub = re.split(r"(?<=[.!?])\s+", part)
            raw_parts.extend(sub)

        results: List[str] = []
        seen: set = set()

        for part in raw_parts:
            cleaned = self._clean_sentence(part)
            if self._is_noise(cleaned):
                continue
            alpha_words = [w for w in cleaned.split() if re.match(r"[A-Za-z]", w)]
            if len(cleaned) < 12 or len(alpha_words) < 3:
                continue
            norm = cleaned[:40].lower()
            if norm in seen:
                continue
            seen.add(norm)
            results.append(cleaned)

        return results

    # ── PROFILE / PERSON query handler ────────────────────────────────────────

    def _answer_person_query(
        self,
        query: str,
        retrieved_chunks: List[SearchResult],
    ) -> str:
        """Frame a clean profile overview for queries like 'who is X' / 'about X'."""
        all_sentences: List[Dict] = []
        person_name: Optional[str] = None

        # Signals that a sentence belongs to the "profile / summary" section
        PROFILE_SIGNALS = [
            "computer science", "engineering student", "b.tech", "btech",
            "hands-on experience", "skilled in", "experienced in", "seeking",
            "data analytics", "business intelligence", "machine learning",
            "artificial intelligence", "internship", "entry-level",
            "analyst", "developer", "engineer",
        ]
        EDUCATION_SIGNALS = [
            "bachelor", "b.tech", "university", "college", "haridwar",
            "gurukul", "vishwavidyalaya", "cgpa", "semester",
        ]
        EXPERIENCE_SIGNALS = [
            "trf limited", "tata enterprise", "intern", "data analytics intern",
            "software engineering intern", "june", "july", "ibm",
        ]

        primary_source = retrieved_chunks[0].chunk.source if retrieved_chunks else "the document"
        primary_page   = retrieved_chunks[0].chunk.page   if retrieved_chunks else 1

        profile_sents: List[str] = []
        education_sents: List[str] = []
        experience_sents: List[str] = []
        other_sents: List[str] = []

        for res in retrieved_chunks:
            chunk = res.chunk
            sentences = self._split_sentences(chunk.chunk_text)

            # Try to extract the candidate's full name from all-caps line
            if person_name is None:
                for line in chunk.chunk_text.split("\n"):
                    stripped = line.strip()
                    # Full-name heuristic: 2-4 words, all alphabetic, all caps
                    words = stripped.split()
                    if (2 <= len(words) <= 4
                            and all(w.isalpha() and w.isupper() for w in words)):
                        person_name = stripped.title()
                        break

            for sent in sentences:
                sl = sent.lower()
                if any(sig in sl for sig in PROFILE_SIGNALS):
                    profile_sents.append(sent)
                elif any(sig in sl for sig in EDUCATION_SIGNALS):
                    education_sents.append(sent)
                elif any(sig in sl for sig in EXPERIENCE_SIGNALS):
                    experience_sents.append(sent)
                else:
                    other_sents.append(sent)

        # Fallback name from query tokens if not found in document
        if person_name is None:
            # Try to extract the name from the query preserving order
            q_lower   = query.lower()
            name_match = re.search(
                r"(?:who\s+is|about|profile\s+of)\s+([A-Za-z]+(?:\s+[A-Za-z]+){0,3})",
                query, re.IGNORECASE
            )
            if name_match:
                person_name = name_match.group(1).title()

        # Build response sections
        lines: List[str] = []

        if person_name:
            lines.append(f"**{person_name}**")

        # Profile paragraph (pick best 1-2 sentences that are long and descriptive)
        profile_selected = _pick_top(profile_sents, max_items=2, min_words=12)
        if profile_selected:
            lines.append("")
            lines.append("**Profile:**")
            for s in profile_selected:
                lines.append(_ensure_period(s))

        # Education
        edu_selected = _pick_top(education_sents, max_items=1, min_words=5)
        if edu_selected:
            lines.append("")
            lines.append("**Education:**")
            for s in edu_selected:
                lines.append(_ensure_period(s))

        # Experience
        exp_selected = _pick_top(experience_sents, max_items=1, min_words=5)
        if exp_selected:
            lines.append("")
            lines.append("**Internship Experience:**")
            for s in exp_selected:
                lines.append(_ensure_period(s))

        # If we found nothing useful, fall back to top sentences overall
        if len(lines) <= 1:
            fallback = _pick_top(other_sents + profile_sents, max_items=3, min_words=8)
            if not fallback:
                return settings.ABSTENTION_MESSAGE
            for s in fallback:
                lines.append(_ensure_period(s))

        lines.append("")
        lines.append(f"[Source: {primary_source}, Page: {primary_page}]")

        return "\n".join(lines)

    # ── SKILLS query handler ──────────────────────────────────────────────────

    def _answer_skills_query(
        self,
        retrieved_chunks: List[SearchResult],
    ) -> str:
        """Extract and display technical skills as a clean bullet list."""
        primary_source = retrieved_chunks[0].chunk.source if retrieved_chunks else "the document"
        primary_page   = retrieved_chunks[0].chunk.page   if retrieved_chunks else 1

        SKILL_SIGNALS = [
            "python", "sql", "java", "power bi", "dax", "pandas", "numpy",
            "scikit", "matplotlib", "seaborn", "plotly", "streamlit",
            "git", "github", "mysql", "postgresql", "excel",
            "machine learning", "deep learning", "nlp", "etl",
            "programming", "querying", "analytics", "data engineering",
            "development", "tools", "frameworks",
        ]

        skill_lines: List[str] = []
        seen_norms: set = set()

        for res in retrieved_chunks:
            sentences = self._split_sentences(res.chunk.chunk_text)
            for sent in sentences:
                sl = sent.lower()
                if any(sig in sl for sig in SKILL_SIGNALS):
                    norm = sent[:35].lower()
                    if norm not in seen_norms:
                        seen_norms.add(norm)
                        skill_lines.append(sent)
                if len(skill_lines) >= 6:
                    break
            if len(skill_lines) >= 6:
                break

        if not skill_lines:
            return settings.ABSTENTION_MESSAGE

        bullets = "\n".join(f"• {s}" for s in skill_lines)
        return (
            f"**Technical Skills:**\n\n"
            f"{bullets}\n\n"
            f"[Source: {primary_source}, Page: {primary_page}]"
        )

    # ── PROJECTS query handler ────────────────────────────────────────────────

    def _answer_projects_query(
        self,
        retrieved_chunks: List[SearchResult],
    ) -> str:
        """Extract project titles and descriptions as a clean bullet list."""
        primary_source = retrieved_chunks[0].chunk.source if retrieved_chunks else "the document"
        primary_page   = retrieved_chunks[0].chunk.page   if retrieved_chunks else 1

        PROJECT_SIGNALS = [
            "built", "developed", "designed", "created", "implemented",
            "analyzed", "automated", "performed", "integrated",
            "dashboard", "system", "portal", "pipeline", "model",
        ]

        project_items: List[str] = []
        seen_norms: set = set()

        for res in retrieved_chunks:
            sentences = self._split_sentences(res.chunk.chunk_text)
            for sent in sentences:
                sl = sent.lower()
                if any(sig in sl for sig in PROJECT_SIGNALS):
                    norm = sent[:35].lower()
                    if norm not in seen_norms:
                        seen_norms.add(norm)
                        project_items.append(sent)
                if len(project_items) >= 6:
                    break
            if len(project_items) >= 6:
                break

        if not project_items:
            return settings.ABSTENTION_MESSAGE

        bullets = "\n".join(f"• {_ensure_period(s)}" for s in project_items)
        return (
            f"**Projects:**\n\n"
            f"{bullets}\n\n"
            f"[Source: {primary_source}, Page: {primary_page}]"
        )

    # ── EDUCATION query handler ───────────────────────────────────────────────

    def _answer_education_query(
        self,
        retrieved_chunks: List[SearchResult],
    ) -> str:
        """Extract education / certification details."""
        primary_source = retrieved_chunks[0].chunk.source if retrieved_chunks else "the document"
        primary_page   = retrieved_chunks[0].chunk.page   if retrieved_chunks else 1

        EDU_SIGNALS = [
            "bachelor", "b.tech", "university", "college", "haridwar",
            "gurukul", "vishwavidyalaya", "cgpa", "semester", "oracle",
            "nptel", "ibm", "certification", "professional", "elite",
        ]

        edu_lines: List[str] = []
        seen_norms: set = set()

        for res in retrieved_chunks:
            sentences = self._split_sentences(res.chunk.chunk_text)
            for sent in sentences:
                sl = sent.lower()
                if any(sig in sl for sig in EDU_SIGNALS):
                    norm = sent[:35].lower()
                    if norm not in seen_norms:
                        seen_norms.add(norm)
                        edu_lines.append(sent)
                if len(edu_lines) >= 5:
                    break
            if len(edu_lines) >= 5:
                break

        if not edu_lines:
            return settings.ABSTENTION_MESSAGE

        bullets = "\n".join(f"• {_ensure_period(s)}" for s in edu_lines)
        return (
            f"**Education & Certifications:**\n\n"
            f"{bullets}\n\n"
            f"[Source: {primary_source}, Page: {primary_page}]"
        )

    # ── EXPERIENCE query handler ──────────────────────────────────────────────

    def _answer_experience_query(
        self,
        retrieved_chunks: List[SearchResult],
    ) -> str:
        """Extract internship / work experience details."""
        primary_source = retrieved_chunks[0].chunk.source if retrieved_chunks else "the document"
        primary_page   = retrieved_chunks[0].chunk.page   if retrieved_chunks else 1

        EXP_SIGNALS = [
            "trf", "tata", "intern", "internship", "june", "july", "ibm",
            "data analytics", "software engineering", "jamshedpur",
        ]

        exp_lines: List[str] = []
        seen_norms: set = set()

        for res in retrieved_chunks:
            sentences = self._split_sentences(res.chunk.chunk_text)
            for sent in sentences:
                sl = sent.lower()
                if any(sig in sl for sig in EXP_SIGNALS):
                    norm = sent[:35].lower()
                    if norm not in seen_norms:
                        seen_norms.add(norm)
                        exp_lines.append(sent)
                if len(exp_lines) >= 4:
                    break
            if len(exp_lines) >= 4:
                break

        if not exp_lines:
            return settings.ABSTENTION_MESSAGE

        bullets = "\n".join(f"• {_ensure_period(s)}" for s in exp_lines)
        return (
            f"**Internship / Work Experience:**\n\n"
            f"{bullets}\n\n"
            f"[Source: {primary_source}, Page: {primary_page}]"
        )

    # ── Generic extractive fallback ───────────────────────────────────────────

    def _extractive_fallback(
        self,
        query: str,
        retrieved_chunks: List[SearchResult],
    ) -> str:
        """Route to a specialized handler based on query intent, or run generic scoring."""
        top_score = retrieved_chunks[0].score if retrieved_chunks else 0.0
        if top_score < settings.MIN_SIMILARITY_THRESHOLD:
            return settings.ABSTENTION_MESSAGE

        intent = self._detect_intent(query)
        logger.debug(f"Extractive intent: {intent}")

        if intent == "person":
            return self._answer_person_query(query, retrieved_chunks)
        if intent == "skills":
            return self._answer_skills_query(retrieved_chunks)
        if intent == "projects":
            return self._answer_projects_query(retrieved_chunks)
        if intent == "education":
            return self._answer_education_query(retrieved_chunks)
        if intent == "experience":
            return self._answer_experience_query(retrieved_chunks)

        # ── Generic factual query: sentence-scoring approach ─────────────────
        return self._generic_factual(query, retrieved_chunks)

    def _generic_factual(
        self,
        query: str,
        retrieved_chunks: List[SearchResult],
    ) -> str:
        """Score sentences by token overlap with the query and return the best ones."""
        raw_tokens    = set(re.findall(r"\b[a-zA-Z0-9_\-\.]+\b", query.lower()))
        content_tokens = raw_tokens - GroundingValidator.STOPWORDS
        q_tokens       = raw_tokens  - META_QUERY_WORDS

        # If no specific content words remain, treat as summary
        if not q_tokens:
            return self._extractive_summary(retrieved_chunks)

        is_numeric = any(w in query.lower() for w in [
            "how many", "how much", "rate", "percentage", "%", "number",
            "score", "days", "hours", "stipend", "cgpa",
        ])

        candidates: List[Dict] = []

        for res in retrieved_chunks:
            sentences = self._split_sentences(res.chunk.chunk_text)
            for sent in sentences:
                sent_tokens = set(re.findall(r"\b[a-zA-Z0-9_\-\.]+\b", sent.lower()))
                overlap = len(q_tokens & sent_tokens)
                if overlap == 0:
                    continue
                score = (overlap / len(q_tokens)) * 0.7 + max(0.0, res.score) * 0.3
                if is_numeric and re.search(r"\d+", sent):
                    score += 0.25
                if len(sent.split()) >= 12:
                    score += 0.20
                candidates.append({
                    "sentence": sent,
                    "score":    score,
                    "overlap":  overlap,
                    "source":   res.chunk.source,
                    "page":     res.chunk.page,
                })

        if not candidates:
            return settings.ABSTENTION_MESSAGE

        candidates.sort(key=lambda x: x["score"], reverse=True)

        # Need at least 0.5 overlap ratio to be confident
        if candidates[0]["overlap"] < 0.5 * len(q_tokens):
            return settings.ABSTENTION_MESSAGE

        selected: List[Dict] = []
        seen_norms: set = set()
        for cand in candidates[:8]:
            norm = cand["sentence"][:35].lower()
            if norm not in seen_norms:
                seen_norms.add(norm)
                selected.append(cand)
            if len(selected) >= 2:
                break

        if not selected:
            return settings.ABSTENTION_MESSAGE

        parts = [_ensure_period(s["sentence"]) for s in selected]
        answer = " ".join(parts)
        source = selected[0]["source"]
        page   = selected[0]["page"]

        return f"{answer} [Source: {source}, Page: {page}]"

    # ── Executive summary ─────────────────────────────────────────────────────

    def _extractive_summary(self, retrieved_chunks: List[SearchResult]) -> str:
        """Build a structured bullet-point document summary."""
        if not retrieved_chunks:
            return settings.ABSTENTION_MESSAGE

        bullets: List[str] = []
        seen: set = set()
        primary_source = retrieved_chunks[0].chunk.source
        primary_page   = retrieved_chunks[0].chunk.page

        for res in retrieved_chunks:
            sentences = self._split_sentences(res.chunk.chunk_text)
            for sent in sentences:
                norm = sent[:35].lower()
                if norm not in seen:
                    seen.add(norm)
                    bullets.append(f"• {_ensure_period(sent)}")
                if len(bullets) >= 6:
                    break
            if len(bullets) >= 6:
                break

        if not bullets:
            return settings.ABSTENTION_MESSAGE

        body = "\n".join(bullets)
        return (
            f"### Document Overview\n\n"
            f"{body}\n\n"
            f"[Source: {primary_source}, Page: {primary_page}]"
        )

    # ── Abstention helper ─────────────────────────────────────────────────────

    @staticmethod
    def _abstain(query: str, note: str) -> "GeneratedAnswer":
        return GeneratedAnswer(
            question=query,
            answer=settings.ABSTENTION_MESSAGE,
            citations=[],
            sources_used=[],
            is_abstained=True,
            groundedness_score=1.0,
            generation_mode="abstention",
            retrieved_chunks=[],
            validation_notes=[note],
        )


# ─────────────────────────────────────────────────────────────────────────────
# Module-level utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_period(text: str) -> str:
    """Append a period if the sentence has no terminal punctuation."""
    t = text.rstrip()
    return t if t.endswith((".", "!", "?")) else t + "."


def _pick_top(sentences: List[str], max_items: int = 2, min_words: int = 8) -> List[str]:
    """Return up to *max_items* longest sentences that meet the *min_words* threshold."""
    eligible = [s for s in sentences if len(s.split()) >= min_words]
    # Prefer longer sentences (more informative)
    eligible.sort(key=lambda s: len(s), reverse=True)
    seen: set = set()
    result: List[str] = []
    for s in eligible:
        norm = s[:35].lower()
        if norm not in seen:
            seen.add(norm)
            result.append(s)
        if len(result) >= max_items:
            break
    return result
