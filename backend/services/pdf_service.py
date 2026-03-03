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

from backend.services.storage_service import review_md_path, review_pdf_path

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


def _tex_escape(text: str) -> str:
    """Escape LaTeX special characters while preserving intended formatting."""
    # First handle backslash (must be done before other replacements)
    text = text.replace("\\", r"\textbackslash{}")
    return _LATEX_ESCAPE_RE.sub(lambda m: _LATEX_ESCAPE_MAP[m.group()], text)


def _tex_escape_url(url: str) -> str:
    """Escape a URL for use in \\href — only escape characters that break LaTeX."""
    return url.replace("#", r"\#").replace("%", r"\%").replace("&", r"\&")


def _tex_escape_with_links(text: str) -> str:
    """Escape LaTeX special chars in text while converting [text](url) to \\href."""
    link_pattern = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')
    parts = []
    last_end = 0
    for m in link_pattern.finditer(text):
        # Escape the text before this link
        parts.append(_tex_escape(text[last_end:m.start()]))
        # Convert the link
        parts.append(r"\href{" + _tex_escape_url(m.group(2)) + "}{" + _tex_escape(m.group(1)) + "}")
        last_end = m.end()
    # Escape the remaining text after the last link
    parts.append(_tex_escape(text[last_end:]))
    return "".join(parts)


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
            # Split into Quote blocks: each starts with "* Quote:"
            quote_blocks = re.split(r"(?=\*\s*Quote\s*:)", ev_text)
            for block in quote_blocks:
                block = block.strip()
                if not block:
                    continue
                q_match = re.match(r"\*\s*Quote\s*:\s*([\s\S]*?)(?=\n\s+\*\s*Comment\s*:|$)", block, re.IGNORECASE)
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
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{microtype}

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
        for cite in parsed.citations:
            cite_text = re.sub(r"^\[\d+\]\s*", "", cite)
            parts.append(r"  \item {\small " + _tex_escape_with_links(cite_text) + "}\n")
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


def _fallback_weasyprint(md_text: str, pdf_path: str) -> str | None:
    """Fallback PDF generation using weasyprint."""
    try:
        import weasyprint

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

    # Fallback
    result = _fallback_weasyprint(md_text, pdf_path)
    if result:
        logger.info("Weasyprint PDF generated for key=%s at %s", key, pdf_path)
    return result
