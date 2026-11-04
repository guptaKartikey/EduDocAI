"""Generate a formal 1-page Candidate Submission Letter PDF using Edge/Chrome headless."""
import os
import sys
import base64
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HTML_PATH = PROJECT_ROOT / "reports" / "submission_letter.html"
PDF_PATH = PROJECT_ROOT / "Submission_Letter.pdf"
DESKTOP_PDF_PATH = Path(r"C:\Users\Kartik\Desktop\Submission_Letter.pdf")
LOGO_PATH = PROJECT_ROOT / "assets" / "logo.png"

logo_img_tag = ""
if LOGO_PATH.exists():
    with open(LOGO_PATH, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    logo_img_tag = f'<img src="data:image/png;base64,{b64}" style="height:55px;object-fit:contain;" alt="EduDocAI">'

html_content = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>AI Intern Submission — Kartikey Gupta</title>
<style>
    @page {{
        size: A4;
        margin: 18mm 18mm 18mm 18mm;
    }}
    body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        color: #1F2937;
        line-height: 1.55;
        font-size: 10pt;
        margin: 0;
        padding: 0;
    }}
    .header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 2.5px solid #2563EB;
        padding-bottom: 12px;
        margin-bottom: 18px;
    }}
    .header-text h1 {{
        margin: 0;
        font-size: 16pt;
        color: #1E3A8A;
        font-weight: 800;
        letter-spacing: -0.3px;
    }}
    .header-text .sub {{
        font-size: 9.5pt;
        color: #4B5563;
        font-weight: 600;
        margin-top: 3px;
    }}
    .subject-box {{
        background: #EFF6FF;
        border-left: 4px solid #2563EB;
        padding: 10px 14px;
        border-radius: 0 6px 6px 0;
        margin-bottom: 16px;
    }}
    .subject-title {{
        font-size: 8.5pt;
        color: #6B7280;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    .subject-text {{
        font-size: 11pt;
        color: #1E40AF;
        font-weight: 700;
        margin-top: 2px;
    }}
    .salutation {{
        font-size: 10.5pt;
        font-weight: 600;
        color: #111827;
        margin-bottom: 8px;
    }}
    p {{
        margin: 0 0 10pt 0;
        text-align: justify;
    }}
    .links-card {{
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 12pt 0;
    }}
    .links-card .card-title {{
        font-size: 9pt;
        font-weight: 700;
        color: #1E3A8A;
        text-transform: uppercase;
        margin-bottom: 8px;
        letter-spacing: 0.4px;
    }}
    .link-row {{
        display: flex;
        align-items: center;
        margin-bottom: 6px;
        font-size: 9.5pt;
    }}
    .link-row:last-child {{
        margin-bottom: 0;
    }}
    .link-label {{
        width: 190px;
        font-weight: 600;
        color: #374151;
    }}
    .link-val a {{
        color: #2563EB;
        text-decoration: none;
        font-weight: 600;
    }}
    .deliverables-box {{
        margin: 12pt 0;
    }}
    .deliverables-title {{
        font-size: 10.5pt;
        font-weight: 700;
        color: #1E40AF;
        margin-bottom: 6pt;
        border-bottom: 1px solid #E5E7EB;
        padding-bottom: 3px;
    }}
    ol {{
        margin: 0;
        padding-left: 18px;
    }}
    li {{
        margin-bottom: 5pt;
        font-size: 9.5pt;
    }}
    li strong {{
        color: #1E3A8A;
    }}
    .declaration-card {{
        background: #F0FDF4;
        border: 1px solid #BBF7D0;
        border-left: 4px solid #16A34A;
        border-radius: 0 6px 6px 0;
        padding: 10px 14px;
        margin: 14pt 0 16pt 0;
    }}
    .declaration-card .dec-title {{
        font-size: 9pt;
        font-weight: 700;
        color: #15803D;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 3px;
    }}
    .declaration-card p {{
        margin: 0;
        font-size: 9pt;
        color: #166534;
        line-height: 1.45;
    }}
    .signoff {{
        margin-top: 14pt;
        display: flex;
        justify-content: space-between;
        align-items: flex-end;
    }}
    .candidate-info {{
        font-size: 9.5pt;
        line-height: 1.5;
    }}
    .candidate-name {{
        font-size: 11.5pt;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 2px;
    }}
    .signature-area {{
        text-align: right;
        font-size: 8.5pt;
        color: #6B7280;
    }}
    .signature-line {{
        width: 180px;
        border-top: 1px solid #9CA3AF;
        margin-bottom: 4px;
    }}
</style>
</head>
<body>

<div class="header">
    <div class="header-text">
        <h1>EduDocAI &mdash; Technical Task Assignment</h1>
        <div class="sub">Candidate Submission Document | Role: AI Engineering & Data Science Intern</div>
    </div>
    {logo_img_tag}
</div>

<div class="subject-box">
    <div class="subject-title">Subject</div>
    <div class="subject-text">AI Engineering & Data Science Intern Submission &mdash; Kartikey Gupta (Track C)</div>
</div>

<div class="salutation">Dear Hiring Team,</div>

<p>
I have completed the technical assessment for the <strong>AI Intern position (Track C: NLP / Generative AI &mdash; Project Concierge)</strong>. 
The objective of this assignment&mdash;to demonstrate engineering judgement, empirical evaluation, and an end-to-end reproducible data-to-decision system&mdash;has been fully satisfied with the implementation of <strong>EduDocAI</strong>.
</p>

<div class="links-card">
    <div class="card-title">Live Verification & Codebase Links</div>
    <div class="link-row">
        <span class="link-label">🔗 Live Application Demo:</span>
        <span class="link-val"><a href="https://edudocai.streamlit.app/">https://edudocai.streamlit.app/</a></span>
    </div>
    <div class="link-row">
        <span class="link-label">🐙 GitHub Repository:</span>
        <span class="link-val"><a href="https://github.com/guptaKartikey/EduDocAI">https://github.com/guptaKartikey/EduDocAI</a></span>
    </div>
</div>

<div class="deliverables-box">
    <div class="deliverables-title">Summary of Deliverables Included</div>
    <ol>
        <li><strong>Video Walkthrough:</strong> 5-minute end-to-end screen demonstration showing dynamic PDF ingestion, factual Q&A, and citation verification.</li>
        <li><strong>Technical Report (PDF):</strong> Comprehensive 10-page report covering problem formulation, RAG architecture, baseline comparison, daily execution logs, and responsible AI.</li>
        <li><strong>FAILURES.md:</strong> Honest engineering failure log documenting 9 real technical bottlenecks, root causes, and resolutions.</li>
        <li><strong>Founder Decision Memo:</strong> Plain-language operational assessment, failure modes, and rollout recommendations.</li>
        <li><strong>Evaluation Benchmark:</strong> 22-question curated evaluation showing 100% Recall@3, 0.9318 MRR, and 90.91% citation accuracy.</li>
    </ol>
</div>

<div class="declaration-card">
    <div class="dec-title">Candidate Declaration</div>
    <p>
    I confirm that this project represents my own original engineering work. All design decisions, code implementations, and failure reflections are documented honestly and ready to be defended in the technical interview.
    </p>
</div>

<div class="signoff">
    <div class="candidate-info">
        <div class="candidate-name">Kartikey Gupta</div>
        <div>B.Tech Computer Science &amp; Engineering</div>
        <div>Email: <a href="mailto:k78048101@gmail.com" style="color:#2563EB;text-decoration:none;">k78048101@gmail.com</a></div>
        <div>LinkedIn: <a href="https://linkedin.com/in/kartikey-gupta-988206372" style="color:#2563EB;text-decoration:none;">linkedin.com/in/kartikey-gupta-988206372</a></div>
    </div>
    <div class="signature-area">
        <div class="signature-line"></div>
        <div>Kartikey Gupta (Candidate)</div>
        <div>Date: November 2026</div>
    </div>
</div>

</body>
</html>
"""

HTML_PATH.write_text(html_content, encoding="utf-8")
print(f"HTML generated at: {HTML_PATH}")

browsers = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
]

chosen = next((b for b in browsers if os.path.exists(b)), None)
if not chosen:
    print("Browser not found")
    sys.exit(1)

cmd = [
    chosen,
    "--headless=new",
    "--disable-gpu",
    "--no-pdf-header-footer",
    f"--print-to-pdf={PDF_PATH}",
    str(HTML_PATH.resolve()),
]

print(f"Running headless browser: {chosen}...")
subprocess.run(cmd, timeout=30)

if PDF_PATH.exists():
    import shutil
    shutil.copy(PDF_PATH, DESKTOP_PDF_PATH)
    print(f"SUCCESS: PDF created at: {PDF_PATH} and copied to {DESKTOP_PDF_PATH}")
else:
    print("Failed to generate PDF")
