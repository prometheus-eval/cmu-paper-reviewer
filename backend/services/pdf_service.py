"""Convert review markdown to PDF using LaTeX (pdflatex).

Falls back to weasyprint if LaTeX compilation fails.
"""

import logging
import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from dataclasses import dataclass, field

import markdown

from backend.services.storage_service import review_md_path, review_pdf_path, verification_code_dir

logger = logging.getLogger(__name__)


# ─── LaTeX special-character escaping ────────────────────────────────────────

_LATEX_ESCAPE_MAP = {
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}
_LATEX_ESCAPE_RE = re.compile("|".join(re.escape(k) for k in _LATEX_ESCAPE_MAP))


_UNICODE_MAP = {
    "\u2014": "---",        # em-dash
    "\u2013": "--",         # en-dash
    "\u2018": "`",          # left single quote
    "\u2019": "'",          # right single quote
    "\u201c": "``",         # left double quote
    "\u201d": "''",         # right double quote
    "\u2026": r"\ldots{}",  # ellipsis
    "\u00a9": r"\textcopyright{}",  # copyright
    "\u00ae": r"\textregistered{}",  # registered
    "\u2122": r"\texttrademark{}",   # trademark
    "\u00b0": r"\textdegree{}",      # degree
    "\u00d7": r"\texttimes{}",       # multiplication sign
    "\u2022": r"\textbullet{}",      # bullet
}


def _tex_escape(text: str) -> str:
    """Escape LaTeX special characters while preserving intended formatting."""
    # First handle backslash (must be done before other replacements)
    text = text.replace("\\", r"\textbackslash{}")
    text = _LATEX_ESCAPE_RE.sub(lambda m: _LATEX_ESCAPE_MAP[m.group()], text)
    # Handle common Unicode characters
    for char, replacement in _UNICODE_MAP.items():
        text = text.replace(char, replacement)
    return text


def _tex_escape_preserving_math(text: str) -> str:
    """Escape LaTeX special characters but preserve $...$ and $$...$$ math blocks."""
    # Extract math blocks before escaping
    math_placeholders = []

    def _save_math(m):
        idx = len(math_placeholders)
        math_placeholders.append(m.group(0))
        return f"%%MATH_PH_{idx}%%"

    # Protect display math $$...$$ first, then inline $...$
    text = re.sub(r'\$\$[\s\S]+?\$\$', _save_math, text)
    text = re.sub(r'\$[^\$\n]+?\$', _save_math, text)

    # Now escape everything else
    text = _tex_escape(text)

    # Restore math blocks (unescaped)
    for idx, math in enumerate(math_placeholders):
        text = text.replace(f"%%MATH\\_PH\\_{idx}%%", math)
        # Also handle case where underscores weren't escaped (shouldn't happen but safe)
        text = text.replace(f"%%MATH_PH_{idx}%%", math)

    return text


def _tex_escape_url(url: str) -> str:
    """Escape a URL for use in \\href — only escape characters that break LaTeX."""
    return url.replace("#", r"\#").replace("%", r"\%").replace("&", r"\&")


def _tex_escape_with_links(text: str, auto_link_citations: bool = True) -> str:
    """Escape LaTeX special chars in text while converting [text](url) to \\href.

    Preserves $...$ and $$...$$ math blocks, and handles citation links.
    """
    # 1. Extract math blocks first
    math_placeholders = []

    def _save_math(m):
        idx = len(math_placeholders)
        math_placeholders.append(m.group(0))
        return f"\x00MATH{idx}\x00"

    text = re.sub(r'\$\$[\s\S]+?\$\$', _save_math, text)
    text = re.sub(r'\$[^\$\n]+?\$', _save_math, text)

    # 2. Handle links and escape non-math text
    link_pattern = re.compile(
        r'\[\[(\d+)\]\]\(([^)]+)\)'   # [[N]](url) — citation reference
        r'|'
        r'\[([^\]]+)\]\(([^)]+)\)'    # [text](url) — standard link
    )
    parts = []
    last_end = 0
    for m in link_pattern.finditer(text):
        parts.append(_tex_escape(text[last_end:m.start()]))
        if m.group(1) is not None:
            num = m.group(1)
            parts.append(r"\hyperlink{ref" + num + r"}{[" + num + r"]}")
        else:
            link_text = m.group(3)
            link_url = m.group(4)
            parts.append(r"\href{" + _tex_escape_url(link_url) + "}{" + _tex_escape(link_text) + "}")
        last_end = m.end()
    parts.append(_tex_escape(text[last_end:]))
    result = "".join(parts)

    # Auto-link plain [N] citation references not already handled
    if auto_link_citations:
        result = re.sub(
            r'(?<!\\hyperlink\{ref)(?<!\[)\[(\d+)\](?!\()',
            lambda m: r"\hyperlink{ref" + m.group(1) + r"}{[" + m.group(1) + r"]}",
            result,
        )

    # 3. Restore math blocks (unescaped)
    for idx, math in enumerate(math_placeholders):
        # The placeholder null bytes survive _tex_escape unchanged
        result = result.replace(f"\x00MATH{idx}\x00", math)

    return result


# ─── Markdown → structured data ─────────────────────────────────────────────

@dataclass
class QuoteComment:
    quote: str
    comment: str


@dataclass
class ReviewItem:
    number: int
    title: str
    main_criticism: str = ""
    eval_criteria: str = ""
    evidence: list[QuoteComment] = field(default_factory=list)


@dataclass
class ParsedReview:
    items: list[ReviewItem] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    raw: str = ""


def _parse_review(md: str) -> ParsedReview:
    """Parse review markdown into structured data."""
    result = ParsedReview(raw=md)

    # Extract citation list
    citation_match = re.search(
        r"^#{1,4}\s*(?:Citation\s*List|References|Citations|Cited Papers)[^\n]*\n([\s\S]*)",
        md, re.MULTILINE | re.IGNORECASE,
    )
    if citation_match:
        for line in citation_match.group(1).strip().split("\n"):
            line = line.strip()
            if line:
                result.citations.append(line)

    # Extract items
    item_pattern = re.compile(r"^##\s*Item\s+(\d+)\s*:\s*(.+)", re.MULTILINE | re.IGNORECASE)
    sections = [(m.group(1), m.group(2).strip(), m.start()) for m in item_pattern.finditer(md)]

    for i, (num, title, start) in enumerate(sections):
        end = sections[i + 1][2] if i + 1 < len(sections) else len(md)
        body = md[start:end]

        item = ReviewItem(number=int(num), title=title)

        # Parse Claim section
        claim_match = re.search(r"####\s*Claim\s*\n([\s\S]*?)(?=####|$)", body, re.IGNORECASE)
        if claim_match:
            claim_text = claim_match.group(1)
            crit_match = re.search(r"\*\s*Main point of criticism\s*:\s*(.+)", claim_text, re.IGNORECASE)
            if crit_match:
                item.main_criticism = crit_match.group(1).strip()
            eval_match = re.search(r"\*\s*Evaluation criteria\s*:\s*(.+)", claim_text, re.IGNORECASE)
            if eval_match:
                item.eval_criteria = eval_match.group(1).strip()

        # Parse Evidence section — extract Quote/Comment pairs
        evidence_match = re.search(r"####\s*Evidence\s*\n([\s\S]*?)(?=####|##\s|$)", body, re.IGNORECASE)
        if evidence_match:
            ev_text = evidence_match.group(1)
            # Normalize bold labels: "* **Quote**:" → "* Quote:"
            ev_text = re.sub(r"\*\s*\*\*Quote\*\*", "* Quote", ev_text)
            ev_text = re.sub(r"\*\s*\*\*Comment\*\*", "* Comment", ev_text)
            # Split into Quote blocks: each starts with "* Quote:" or "* Quote from ...:"
            quote_blocks = re.split(r"(?=\*\s*Quote(?:\s+from\s+[^:]+)?\s*:)", ev_text)
            for block in quote_blocks:
                block = block.strip()
                if not block:
                    continue
                # Match quote text up to comment (indented or same level)
                q_match = re.match(r"\*\s*Quote(?:\s+from\s+[^:]+)?\s*:\s*([\s\S]*?)(?=\n\s*\*\s*Comment\s*:|$)", block, re.IGNORECASE)
                c_match = re.search(r"\*\s*Comment\s*:\s*([\s\S]*?)$", block, re.IGNORECASE)
                if q_match:
                    quote_text = q_match.group(1).strip()
                    comment_text = c_match.group(1).strip() if c_match else ""
                    item.evidence.append(QuoteComment(quote=quote_text, comment=comment_text))

        result.items.append(item)

    return result


# ─── LaTeX document generation ───────────────────────────────────────────────

LATEX_PREAMBLE = r"""
\documentclass[11pt,a4paper]{article}

% Encoding & fonts
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{textcomp}
\usepackage{microtype}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{bbm}
\usepackage{bm}

\sloppy

% Page layout
\usepackage[margin=2.2cm, top=2.8cm, bottom=2.8cm]{geometry}

% Colors — matched to web CSS variables
\usepackage{xcolor}
\definecolor{cmured}{HTML}{C41230}
\definecolor{cmureddark}{HTML}{9E0E27}
\definecolor{lightred}{HTML}{FDF2F4}
\definecolor{lightgray}{HTML}{FAFAFA}
\definecolor{midgray}{HTML}{F5F5F5}
\definecolor{darkgray}{HTML}{262626}
\definecolor{medgray}{HTML}{737373}
\definecolor{textgray}{HTML}{404040}
\definecolor{bordergray}{HTML}{E5E5E5}
\definecolor{infoblue}{HTML}{2563EB}
\definecolor{infobluebg}{HTML}{EFF6FF}

% Graphics & drawing
\usepackage{tikz}
\usetikzlibrary{calc}

% Links
\usepackage[colorlinks=true,linkcolor=cmured,urlcolor=cmured,citecolor=cmured]{hyperref}

% Headers & footers
\usepackage{fancyhdr}
\usepackage{lastpage}
\pagestyle{fancy}
\fancyhf{}
\renewcommand{\headrulewidth}{0pt}
\fancyhead[L]{\small\color{medgray}\textsf{CMU Paper Reviewer}}
\fancyhead[R]{\small\color{medgray}\textsf{AI-Generated Review}}
\fancyfoot[C]{\small\color{medgray}\textsf{Page \thepage\ of \pageref{LastPage}}}
% Red top line on every page
\renewcommand{\headrule}{{\color{cmured}\hrule height 2pt\vspace{2pt}}}

% Lists
\usepackage{enumitem}

% Boxes
\usepackage{tcolorbox}
\tcbuselibrary{breakable,skins}

% ── Item Card: outer container that mimics the web card ──
\newtcolorbox{itemcard}[1]{
  enhanced,
  breakable,
  colback=white,
  colframe=bordergray,
  boxrule=0.6pt,
  arc=4pt,
  left=14pt, right=14pt, top=10pt, bottom=10pt,
  title={\Large\bfseries\color{darkgray} #1},
  coltitle=darkgray,
  fonttitle=\sffamily\bfseries\large,
  colbacktitle=white,
  attach boxed title to top left={yshift=-2mm,xshift=4mm},
  boxed title style={colback=white,colframe=white,boxrule=0pt,sharp corners},
  before upper={\vspace{2pt}},
}

% ── Claim box: red left border, pink background ──
\newtcolorbox{claimbox}{
  enhanced,
  colback=lightred,
  colframe=cmured,
  boxrule=0pt,
  leftrule=3.5pt,
  arc=0pt,
  outer arc=2pt,
  breakable,
  top=10pt, bottom=10pt, left=14pt, right=14pt,
  fontupper=\color{textgray},
  before skip=6pt,
  after skip=8pt,
}

% ── Criteria pill ──
\newcommand{\criteriapill}[1]{%
  \tikz[baseline=(pill.base)]{%
    \node[fill=infobluebg, draw=infoblue!30, rounded corners=8pt,
          inner xsep=8pt, inner ysep=3pt, font=\small\sffamily\color{infoblue}]
          (pill) {#1};%
  }%
}

% ── Quote box: gray background, rounded ──
\newtcolorbox{quotebox}[1][]{
  enhanced,
  colback=midgray,
  colframe=bordergray,
  boxrule=0.5pt,
  arc=4pt,
  breakable,
  top=8pt, bottom=8pt, left=12pt, right=12pt,
  fontupper=\color{textgray},
  before skip=4pt,
  after skip=2pt,
  #1
}

% ── Comment box: indented, gray left border ──
\newtcolorbox{commentbox}{
  enhanced,
  colback=white,
  colframe=bordergray,
  boxrule=0pt,
  leftrule=2.5pt,
  arc=0pt,
  breakable,
  top=8pt, bottom=8pt, left=12pt, right=12pt,
  left skip=16pt,
  fontupper=\color{textgray}\small,
  before skip=2pt,
  after skip=6pt,
}

% ── Section-style commands ──
\newcommand{\evidenceheader}{%
  \vspace{6pt}%
  {\small\sffamily\bfseries\color{medgray}\MakeUppercase{Evidence}}%
  \vspace{2pt}%
  {\color{bordergray}\hrule height 0.5pt}%
  \vspace{6pt}%
}

\newcommand{\itemheader}[2]{%
  % #1 = number, #2 = title
  \vspace{1em}%
  \noindent
  \begin{tikzpicture}[baseline=(num.base)]
    \node[circle, fill=lightred, text=cmured, font=\sffamily\bfseries\small,
          minimum size=22pt, inner sep=0pt] (num) {#1};
  \end{tikzpicture}%
  \hspace{6pt}%
  {\Large\sffamily\bfseries\color{darkgray} #2}%
  \vspace{4pt}%
  {\color{cmured}\hrule height 1.5pt}%
  \vspace{8pt}%
}

% Misc
\usepackage{parskip}
\setlength{\parindent}{0pt}
\setlength{\parskip}{0.4em}

\begin{document}
"""

LATEX_END = r"""
\end{document}
"""


def _generate_latex(parsed: ParsedReview, key: str, model_name: str = "") -> str:
    """Generate a LaTeX document string from parsed review data."""
    now = datetime.now(timezone.utc).strftime("%B %d, %Y")
    model_display = model_name.split("/")[-1] if model_name else "AI"

    parts = [LATEX_PREAMBLE]

    # ── Title block ──
    parts.append(r"""
\begin{center}
  {\color{cmured}\rule{\textwidth}{2pt}}\\[1em]
  {\LARGE\sffamily\bfseries AI-Generated Paper Review}\\[0.6em]
  {\small\color{medgray}\sffamily Date: """ + _tex_escape(now) + r""" \quad\textbar\quad Submission Key: \texttt{""" + _tex_escape(key) + r"""} \quad\textbar\quad Model: """ + _tex_escape(model_display) + r"""}\\[0.6em]
  {\color{cmured}\rule{\textwidth}{2pt}}
\end{center}
\vspace{1em}

\noindent
{\sffamily\large\bfseries\color{darkgray} Review Summary}\hspace{8pt}%
{\sffamily\color{medgray} """ + str(len(parsed.items)) + r""" review items \quad\textbar\quad """ + str(len(parsed.citations)) + r""" citations}
\par
\vspace{1.5em}
""")

    # ── Items ──
    for item in parsed.items:
        # Item header with numbered circle
        parts.append(r"\itemheader{" + str(item.number) + "}{" + _tex_escape(item.title) + "}\n\n")

        # Claim box with red label
        parts.append(r"\begin{claimbox}" + "\n")
        parts.append(r"{\sffamily\bfseries\small\color{cmured}\MakeUppercase{Main Point of Criticism}}" + "\n\n")
        if item.main_criticism:
            parts.append(_tex_escape_with_links(item.main_criticism) + "\n")
        parts.append(r"\end{claimbox}" + "\n\n")

        # Evaluation criteria pill
        if item.eval_criteria:
            parts.append(r"\noindent {\small\sffamily\color{medgray} Evaluation Criteria:}\hspace{6pt}")
            parts.append(r"\criteriapill{" + _tex_escape(item.eval_criteria) + "}\n\n")

        # Evidence section
        if item.evidence:
            parts.append(r"\evidenceheader{}" + "\n\n")

            for j, ev in enumerate(item.evidence):
                # Quote box
                parts.append(r"\begin{quotebox}" + "\n")
                parts.append(r"{\sffamily\bfseries\small\color{medgray}\MakeUppercase{")
                parts.append(r"Quote " + str(j + 1) + r"}}" + "\n\n")
                parts.append(r"{\itshape " + _tex_escape_with_links(ev.quote) + "}\n")
                parts.append(r"\end{quotebox}" + "\n")

                # Comment box (indented)
                if ev.comment:
                    parts.append(r"\begin{commentbox}" + "\n")
                    parts.append(r"{\sffamily\bfseries\small\color{medgray}\MakeUppercase{Comment}}" + "\n\n")
                    parts.append(_tex_escape_with_links(ev.comment) + "\n")
                    parts.append(r"\end{commentbox}" + "\n")

                parts.append(r"\vspace{4pt}" + "\n")

        parts.append(r"\vspace{0.8em}" + "\n\n")

    # ── Verification code (before references) ──
    vdir = verification_code_dir(key)
    vcode_files = []
    if vdir.exists():
        vcode_files = sorted(f for f in vdir.rglob("*") if f.is_file())
    if vcode_files:
        parts.append(r"\vspace{0.5em}" + "\n")
        parts.append(r"\noindent")
        parts.append(r"\begin{tikzpicture}[baseline=(num.base)]")
        parts.append(r"  \node[circle, fill=lightred, text=cmured, font=\sffamily\bfseries\small,")
        parts.append(r"        minimum size=22pt, inner sep=0pt] (num) {$\langle/\rangle$};")
        parts.append(r"\end{tikzpicture}")
        parts.append(r"\hspace{6pt}")
        parts.append(r"{\Large\sffamily\bfseries\color{darkgray} Verification Code}")
        parts.append(r"\hspace{8pt}{\sffamily\color{medgray}(" + str(len(vcode_files)) + r" files)}" + "\n")
        parts.append(r"\vspace{4pt}" + "\n")
        parts.append(r"{\color{cmured}\hrule height 1.5pt}" + "\n")
        parts.append(r"\vspace{8pt}" + "\n")
        parts.append(r"{\small\sffamily\color{medgray}\itshape The AI reviewer generated the following code to verify claims in the paper.}" + "\n\n")

        for vf in vcode_files:
            fname = str(vf.relative_to(vdir))
            try:
                content = vf.read_text(encoding="utf-8")
                # Truncate very long files
                if len(content) > 3000:
                    content = content[:3000] + "\n... (truncated)"
            except UnicodeDecodeError:
                content = "(binary file)"
            parts.append(r"\noindent{\small\sffamily\bfseries\color{textgray} " + _tex_escape(fname) + "}\n\n")
            parts.append(r"\begin{quotebox}" + "\n")
            parts.append(r"{\ttfamily\scriptsize " + "\n")
            # Escape and use verbatim-like formatting
            escaped = _tex_escape(content).replace("\n", r"\\" + "\n")
            parts.append(escaped + "\n")
            parts.append(r"}" + "\n")
            parts.append(r"\end{quotebox}" + "\n\n")

    # ── Citation list ──
    if parsed.citations:
        parts.append(r"\vspace{0.5em}" + "\n")
        parts.append(r"\noindent")
        parts.append(r"\begin{tikzpicture}[baseline=(num.base)]")
        parts.append(r"  \node[circle, fill=lightred, text=cmured, font=\sffamily\bfseries\small,")
        parts.append(r"        minimum size=22pt, inner sep=0pt] (num) {\#};")
        parts.append(r"\end{tikzpicture}")
        parts.append(r"\hspace{6pt}")
        parts.append(r"{\Large\sffamily\bfseries\color{darkgray} References}")
        parts.append(r"\hspace{8pt}{\sffamily\color{medgray}(" + str(len(parsed.citations)) + r")}" + "\n")
        parts.append(r"\vspace{4pt}" + "\n")
        parts.append(r"{\color{cmured}\hrule height 1.5pt}" + "\n")
        parts.append(r"\vspace{8pt}" + "\n")
        parts.append(r"\begin{enumerate}[leftmargin=2em, label={\color{cmured}[\arabic*]}, itemsep=0.4em]" + "\n")
        for idx, cite in enumerate(parsed.citations, 1):
            cite_text = re.sub(r"^\[\d+\]\s*", "", cite)
            parts.append(r"  \item \hypertarget{ref" + str(idx) + r"}{}" + r"{\small " + _tex_escape_with_links(cite_text, auto_link_citations=False) + "}\n")
        parts.append(r"\end{enumerate}" + "\n")

    # ── Disclaimer ──
    parts.append(r"""
\vfill
\noindent{\color{bordergray}\hrule height 0.5pt}
\vspace{8pt}
\begin{center}
  \small\sffamily\color{medgray}\itshape
  This review was generated by an AI system and should be used as supplementary feedback only.\\
  It does not replace human expert peer review.
\end{center}
""")

    parts.append(LATEX_END)
    return "".join(parts)


def _compile_latex(tex_content: str, output_path: str) -> bool:
    """Compile LaTeX to PDF using pdflatex. Returns True on success."""
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tex_file = os.path.join(tmpdir, "review.tex")
            with open(tex_file, "w", encoding="utf-8") as f:
                f.write(tex_content)

            # Run pdflatex twice (for page refs like lastpage)
            for run in range(2):
                result = subprocess.run(
                    ["pdflatex", "-interaction=nonstopmode", "-output-directory", tmpdir, tex_file],
                    capture_output=True, text=True, timeout=60,
                )
                if result.returncode != 0 and run == 1:
                    logger.warning("pdflatex run %d failed:\n%s", run + 1, result.stdout[-2000:])
                    return False

            pdf_file = os.path.join(tmpdir, "review.pdf")
            if os.path.exists(pdf_file):
                shutil.copy2(pdf_file, output_path)
                return True
    except FileNotFoundError:
        logger.warning("pdflatex not found on system; falling back to weasyprint")
        return False

    return False


# ─── Fallback: weasyprint ───────────────────────────────────────────────────

WEASYPRINT_CSS = """
@page { size: A4; margin: 2cm; }
body { font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; font-size: 11pt; line-height: 1.6; color: #222; }
h1, h2, h3, h4 { color: #C41230; }
code { background: #f4f4f4; padding: 2px 4px; border-radius: 3px; font-size: 0.9em; }
pre { background: #f4f4f4; padding: 12px; border-radius: 6px; overflow-x: auto; }
blockquote { border-left: 4px solid #C41230; margin: 1em 0; padding: 0.5em 1em; background: #fdf2f4; }
"""

WEASYPRINT_STRUCTURED_CSS = """
@page { size: A4; margin: 2cm; @top-left { content: "CMU Paper Reviewer"; font-size: 9pt; color: #737373; } @top-right { content: "AI-Generated Review"; font-size: 9pt; color: #737373; } @bottom-center { content: "Page " counter(page) " of " counter(pages); font-size: 9pt; color: #737373; } }
body { font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; font-size: 11pt; line-height: 1.6; color: #404040; }
.header-bar { background: #C41230; color: #fff; padding: 12px 20px; border-radius: 6px; margin-bottom: 1.5em; }
.header-bar h1 { color: #fff; font-size: 18pt; margin: 0; }
.header-bar .meta { font-size: 9pt; color: rgba(255,255,255,0.8); margin-top: 4px; }
.summary-line { font-size: 13pt; font-weight: bold; color: #262626; margin-bottom: 0.3em; }
.summary-stats { font-size: 10pt; color: #737373; margin-bottom: 1.5em; }
.item-header { display: flex; align-items: center; gap: 8px; margin-top: 1.5em; margin-bottom: 0.5em; padding-bottom: 6px; border-bottom: 2px solid #C41230; }
.item-num { display: inline-block; width: 24px; height: 24px; line-height: 24px; text-align: center; border-radius: 50%; background: #fdf2f4; color: #C41230; font-weight: bold; font-size: 10pt; }
.item-title { font-size: 14pt; font-weight: bold; color: #262626; }
.claim-box { background: #fdf2f4; border-left: 3px solid #C41230; padding: 10px 14px; border-radius: 0 6px 6px 0; margin: 8px 0; }
.claim-box .label { font-size: 8pt; font-weight: bold; text-transform: uppercase; letter-spacing: 0.05em; color: #C41230; margin-bottom: 4px; }
.criteria-pill { display: inline-block; background: #EFF6FF; color: #2563EB; border: 1px solid #BFDBFE; border-radius: 12px; padding: 2px 10px; font-size: 9pt; font-weight: 600; }
.evidence-header { font-size: 9pt; font-weight: bold; text-transform: uppercase; letter-spacing: 0.05em; color: #737373; margin-top: 12px; margin-bottom: 6px; padding-bottom: 4px; border-bottom: 1px solid #E5E5E5; }
.quote-box { background: #F5F5F5; border: 1px solid #E5E5E5; border-radius: 6px; padding: 8px 12px; margin: 4px 0; }
.quote-box .label { font-size: 8pt; font-weight: bold; text-transform: uppercase; color: #A3A3A3; margin-bottom: 2px; }
.quote-box .text { font-style: italic; color: #404040; }
.comment-box { border-left: 2px solid #D4D4D4; margin-left: 16px; padding: 6px 12px; margin-top: 2px; margin-bottom: 6px; }
.comment-box .label { font-size: 8pt; font-weight: bold; text-transform: uppercase; color: #A3A3A3; margin-bottom: 2px; }
.comment-box .text { font-size: 10pt; color: #404040; }
.references { margin-top: 2em; }
.references h2 { font-size: 14pt; color: #262626; border-bottom: 2px solid #C41230; padding-bottom: 4px; }
.references ol { padding-left: 2em; }
.references li { font-size: 10pt; color: #525252; padding: 3px 0; }
.disclaimer { margin-top: 2em; padding-top: 8px; border-top: 1px solid #E5E5E5; text-align: center; font-size: 9pt; font-style: italic; color: #737373; }
a { color: #C41230; }
"""


def _generate_structured_html(parsed: ParsedReview, key: str, model_name: str = "") -> str:
    """Generate professional HTML from parsed review for weasyprint fallback."""
    now = datetime.now(timezone.utc).strftime("%B %d, %Y")
    model_display = model_name.split("/")[-1] if model_name else "AI"

    def _html_esc(text: str) -> str:
        return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")

    def _md_links(text: str, auto_link_citations: bool = True) -> str:
        """Convert [text](url) and [[N]](#anchor) to <a> tags in otherwise-escaped text."""
        import re as _re
        # Combined pattern: nested-bracket citations OR standard links
        pattern = _re.compile(
            r'\[\[(\d+)\]\]\(([^)]+)\)'   # [[N]](url)
            r'|'
            r'\[([^\]]+)\]\(([^)]+)\)'    # [text](url)
        )
        parts = []
        last = 0
        for m in pattern.finditer(text):
            parts.append(_html_esc(text[last:m.start()]))
            if m.group(1) is not None:
                parts.append(f'<a href="#ref{m.group(1)}">[{m.group(1)}]</a>')
            else:
                parts.append(f'<a href="{_html_esc(m.group(4))}">{_html_esc(m.group(3))}</a>')
            last = m.end()
        parts.append(_html_esc(text[last:]))
        result = "".join(parts)
        # Auto-link plain [N] citation references not already converted
        if auto_link_citations:
            result = _re.sub(
                r'(?<!<a href="#ref)(?<!\[)\[(\d+)\](?!\()',
                lambda m: f'<a href="#ref{m.group(1)}">[{m.group(1)}]</a>',
                result,
            )
        return result

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><style>{WEASYPRINT_STRUCTURED_CSS}</style></head><body>
<div class="header-bar">
  <h1>AI-Generated Paper Review</h1>
  <div class="meta">Date: {_html_esc(now)} &nbsp;|&nbsp; Submission Key: {_html_esc(key)} &nbsp;|&nbsp; Model: {_html_esc(model_display)}</div>
</div>
<div class="summary-line">Review Summary</div>
<div class="summary-stats">{len(parsed.items)} review items &nbsp;|&nbsp; {len(parsed.citations)} citations</div>
"""

    for item in parsed.items:
        html += f'<div class="item-header"><span class="item-num">{item.number}</span><span class="item-title">{_html_esc(item.title)}</span></div>\n'

        if item.main_criticism:
            html += f'<div class="claim-box"><div class="label">Main Point of Criticism</div><div>{_md_links(item.main_criticism)}</div></div>\n'

        if item.eval_criteria:
            html += f'<p style="margin:6px 0;"><span style="font-size:9pt;color:#737373;">Evaluation Criteria:</span> <span class="criteria-pill">{_html_esc(item.eval_criteria)}</span></p>\n'

        if item.evidence:
            html += '<div class="evidence-header">Evidence</div>\n'
            for j, ev in enumerate(item.evidence):
                html += f'<div class="quote-box"><div class="label">Quote {j + 1}</div><div class="text">{_md_links(ev.quote)}</div></div>\n'
                if ev.comment:
                    html += f'<div class="comment-box"><div class="label">Comment</div><div class="text">{_md_links(ev.comment)}</div></div>\n'

    # Verification code
    vdir = verification_code_dir(key)
    vcode_files = []
    if vdir.exists():
        vcode_files = sorted(f for f in vdir.rglob("*") if f.is_file())
    if vcode_files:
        html += '<div class="references"><h2>Verification Code (' + str(len(vcode_files)) + ' files)</h2>\n'
        html += '<p style="font-size:9pt;color:#737373;font-style:italic;">The AI reviewer generated the following code to verify claims in the paper.</p>\n'
        for vf in vcode_files:
            fname = str(vf.relative_to(vdir))
            try:
                content = vf.read_text(encoding="utf-8")
                if len(content) > 3000:
                    content = content[:3000] + "\n... (truncated)"
            except UnicodeDecodeError:
                content = "(binary file)"
            html += f'<p style="font-weight:bold;font-size:10pt;margin:8px 0 2px;">{_html_esc(fname)}</p>\n'
            html += f'<pre style="background:#f5f5f5;border:1px solid #e5e5e5;border-radius:6px;padding:8px;font-size:8pt;overflow-x:auto;white-space:pre-wrap;">{_html_esc(content)}</pre>\n'
        html += '</div>\n'

    if parsed.citations:
        html += '<div class="references"><h2>References (' + str(len(parsed.citations)) + ')</h2><ol>\n'
        for idx, cite in enumerate(parsed.citations, 1):
            cite_text = re.sub(r"^\[\d+\]\s*", "", cite)
            html += f'  <li id="ref{idx}">{_md_links(cite_text, auto_link_citations=False)}</li>\n'
        html += '</ol></div>\n'

    html += '<div class="disclaimer">This review was generated by an AI system and should be used as supplementary feedback only.<br>It does not replace human expert peer review.</div>\n'
    html += '</body></html>'
    return html


def _fallback_weasyprint(md_text: str, pdf_path: str, parsed: ParsedReview | None = None, key: str = "", model_name: str = "") -> str | None:
    """Fallback PDF generation using weasyprint."""
    try:
        import weasyprint

        if parsed and parsed.items:
            full_html = _generate_structured_html(parsed, key, model_name)
        else:
            html_body = markdown.markdown(md_text, extensions=["fenced_code", "tables", "codehilite"])
            full_html = f'<!DOCTYPE html><html><head><meta charset="utf-8"><style>{WEASYPRINT_CSS}</style></head><body>{html_body}</body></html>'
        weasyprint.HTML(string=full_html).write_pdf(str(pdf_path))
        return str(pdf_path)
    except Exception:
        logger.exception("Weasyprint fallback also failed")
        return None


# ─── Public API ──────────────────────────────────────────────────────────────

def generate_review_pdf(key: str, model_name: str = "") -> str | None:
    """Convert review.md to review.pdf using LaTeX. Falls back to weasyprint.

    Returns the PDF path or None if no markdown exists.
    """
    md_path = review_md_path(key)
    if not md_path.exists():
        return None

    md_text = md_path.read_text(encoding="utf-8")
    pdf_path = str(review_pdf_path(key))

    # Try LaTeX first
    parsed = None
    try:
        parsed = _parse_review(md_text)
        if parsed.items:
            tex = _generate_latex(parsed, key, model_name)
            if _compile_latex(tex, pdf_path):
                logger.info("LaTeX PDF generated for key=%s at %s", key, pdf_path)
                return pdf_path
            else:
                logger.warning("LaTeX compilation failed for key=%s, falling back to weasyprint", key)
        else:
            logger.info("No structured items found for key=%s, using weasyprint", key)
    except Exception:
        logger.exception("LaTeX generation failed for key=%s, falling back to weasyprint", key)

    # Fallback — pass parsed data for structured HTML if available
    result = _fallback_weasyprint(md_text, pdf_path, parsed=parsed, key=key, model_name=model_name)
    if result:
        logger.info("Weasyprint PDF generated for key=%s at %s", key, pdf_path)
    return result
