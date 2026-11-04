"""Convert technical_report.md into a professional styled PDF using Edge/Chrome headless."""
import os
import sys
import base64
import subprocess
from pathlib import Path
import markdown

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MD_PATH = PROJECT_ROOT / "reports" / "technical_report.md"
HTML_PATH = PROJECT_ROOT / "reports" / "technical_report.html"
PDF_PATH = PROJECT_ROOT / "Technical_Report.pdf"
LOGO_PATH = PROJECT_ROOT / "assets" / "logo.png"

def build_html():
    with open(MD_PATH, "r", encoding="utf-8") as f:
        md_text = f.read()

    # Convert markdown to html with standard extensions
    html_body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "nl2br", "sane_lists"]
    )

    logo_img_tag = ""
    if LOGO_PATH.exists():
        with open(LOGO_PATH, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        logo_img_tag = f'<img src="data:image/png;base64,{b64}" style="height:55px;object-fit:contain;" alt="EduDocAI">'

    css = """
    @page {
        size: A4;
        margin: 18mm 16mm 18mm 16mm;
    }
    body {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        color: #1F2937;
        line-height: 1.6;
        font-size: 9.8pt;
        margin: 0;
        padding: 0;
        -webkit-font-smoothing: antialiased;
    }
    .top-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        border-bottom: 2.5px solid #2563EB;
        padding-bottom: 12px;
        margin-bottom: 20px;
    }
    .top-header .title-group h1 {
        margin: 0;
        font-size: 18pt;
        color: #1E3A8A;
        font-weight: 800;
        letter-spacing: -0.4px;
    }
    .top-header .title-group .sub {
        font-size: 10pt;
        color: #4B5563;
        font-weight: 600;
        margin-top: 3px;
    }
    .candidate-badge {
        display: inline-block;
        background: #EFF6FF;
        border: 1px solid #BFDBFE;
        color: #1E40AF;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 8.5pt;
        font-weight: 600;
        margin-top: 6px;
    }
    h1 {
        color: #1E3A8A;
        font-size: 18pt;
        margin-top: 0;
        margin-bottom: 8px;
        font-weight: 800;
        letter-spacing: -0.3px;
    }
    h2 {
        color: #1D4ED8;
        font-size: 13.5pt;
        border-bottom: 1.5px solid #DBEAFE;
        padding-bottom: 4px;
        margin-top: 18pt;
        margin-bottom: 8pt;
        font-weight: 700;
        page-break-after: avoid;
    }
    h3 {
        color: #1E40AF;
        font-size: 11pt;
        margin-top: 12pt;
        margin-bottom: 5pt;
        font-weight: 600;
        page-break-after: avoid;
    }
    p {
        margin-top: 0;
        margin-bottom: 8pt;
        text-align: justify;
    }
    table {
        width: 100%;
        border-collapse: collapse;
        margin: 12pt 0;
        font-size: 8.8pt;
        page-break-inside: avoid;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    tr {
        page-break-inside: avoid;
    }
    th, td {
        border: 1px solid #D1D5DB;
        padding: 6px 10px;
        text-align: left;
    }
    th {
        background-color: #EFF6FF;
        color: #1E40AF;
        font-weight: 700;
        font-size: 8.8pt;
    }
    tr:nth-child(even) {
        background-color: #F9FAFB;
    }
    pre {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        padding: 10px 12px;
        font-size: 8pt;
        line-height: 1.4;
        overflow-x: auto;
        font-family: "Consolas", "Courier New", monospace;
        page-break-inside: avoid;
        color: #0F172A;
    }
    code {
        background: #EFF6FF;
        color: #1E40AF;
        padding: 1px 4px;
        border-radius: 3px;
        font-family: "Consolas", "Courier New", monospace;
        font-size: 8.8pt;
    }
    pre code {
        background: transparent;
        padding: 0;
        color: #0F172A;
        font-size: 8pt;
    }
    blockquote {
        border-left: 4px solid #2563EB;
        margin: 10pt 0;
        padding: 8pt 12pt;
        background: #F0F7FF;
        color: #1E3A8A;
        border-radius: 0 6px 6px 0;
    }
    ul, ol {
        margin-top: 4pt;
        margin-bottom: 8pt;
        padding-left: 20pt;
    }
    li {
        margin-bottom: 3pt;
    }
    hr {
        border: none;
        border-top: 1px solid #E5E7EB;
        margin: 16pt 0;
    }
    .footer-stamp {
        margin-top: 26pt;
        border-top: 1px solid #CBD5E1;
        padding-top: 10pt;
        font-size: 8pt;
        color: #64748B;
        text-align: center;
        font-weight: 500;
    }
    """

    full_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>EduRankAI Technical Task Report — Kartikey Gupta</title>
<style>{css}</style>
</head>
<body>

<div class="top-header">
    <div class="title-group">
        <h1>EduDocAI: Technical Task Report</h1>
        <div class="sub">Track C: NLP / Generative AI — Candidate Task Assignment (EduRankAI)</div>
        <div class="candidate-badge">
            Candidate: Kartikey Gupta &nbsp;|&nbsp; B.Tech CSE &nbsp;|&nbsp; k78048101@gmail.com
        </div>
    </div>
    {logo_img_tag}
</div>

{html_body}

<div class="footer-stamp">
    EduDocAI — AI Engineering & Data Science Technical Assessment | Kartikey Gupta
</div>

</body>
</html>
"""

    with open(HTML_PATH, "w", encoding="utf-8") as f:
        f.write(full_html)
    print(f"HTML generated at: {HTML_PATH}")


def convert_to_pdf():
    # Candidates for Chromium / Edge executable
    browsers = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    ]

    chosen = None
    for b in browsers:
        if os.path.exists(b):
            chosen = b
            break

    if not chosen:
        print("No Chromium-based browser found. Falling back to pdf generation...")
        return False

    cmd = [
        chosen,
        "--headless=new",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={PDF_PATH}",
        str(HTML_PATH.resolve()),
    ]
    print(f"Running headless browser: {chosen}...")
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if os.path.exists(PDF_PATH):
        size_kb = os.path.getsize(PDF_PATH) / 1024
        print(f"SUCCESS: PDF created at: {PDF_PATH} ({size_kb:.1f} KB)")
        return True
    else:
        print(f"FAILED to generate PDF via browser: {res.stderr}")
        return False


if __name__ == "__main__":
    build_html()
    success = convert_to_pdf()
    if not success:
        sys.exit(1)
