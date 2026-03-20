"""Extract the publication/submission date of a paper from OCR text or OpenAlex."""

import logging
import re
from datetime import date

import requests

logger = logging.getLogger(__name__)

# ── Regex patterns for common date formats in papers ─────────────────────────

# "Received: 15 January 2025" / "Submitted: March 2025" / "Accepted: 2025-01-15"
_DATE_LABEL_PATTERN = re.compile(
    r"(?:received|submitted|accepted|published|date|revised|posted)"
    r"\s*[:;]?\s*"
    r"(\d{1,2}\s+(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}"
    r"|(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}"
    r"|(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}"
    r"|\d{4}[-/]\d{2}[-/]\d{2}"
    r"|\d{1,2}[-/]\d{1,2}[-/]\d{4})",
    re.IGNORECASE,
)

# arXiv ID pattern: 2501.12345 → January 2025
_ARXIV_ID_PATTERN = re.compile(r"\barXiv:\s*(\d{2})(\d{2})\.\d{4,5}", re.IGNORECASE)

# arXiv URL: arxiv.org/abs/2501.12345
_ARXIV_URL_PATTERN = re.compile(r"arxiv\.org/(?:abs|pdf)/(\d{2})(\d{2})\.\d{4,5}", re.IGNORECASE)

# Copyright year: "© 2025" or "Copyright 2025"
_COPYRIGHT_PATTERN = re.compile(r"(?:©|copyright)\s*(\d{4})", re.IGNORECASE)

# Standalone date in first ~2000 chars: "January 15, 2025" or "15 January 2025"
_STANDALONE_DATE = re.compile(
    r"(\d{1,2}\s+(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}"
    r"|(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4})",
    re.IGNORECASE,
)

_MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}


def _parse_date_str(s: str) -> date | None:
    """Try to parse a date string into a date object."""
    s = s.strip().rstrip(".")

    # ISO: 2025-01-15 or 2025/01/15
    m = re.match(r"(\d{4})[-/](\d{2})[-/](\d{2})", s)
    if m:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))

    # US: 01/15/2025
    m = re.match(r"(\d{1,2})[-/](\d{1,2})[-/](\d{4})", s)
    if m:
        return date(int(m.group(3)), int(m.group(1)), int(m.group(2)))

    # "15 January 2025"
    m = re.match(r"(\d{1,2})\s+(\w+)\s+(\d{4})", s, re.IGNORECASE)
    if m and m.group(2).lower() in _MONTH_MAP:
        return date(int(m.group(3)), _MONTH_MAP[m.group(2).lower()], int(m.group(1)))

    # "January 15, 2025" or "January 2025"
    m = re.match(r"(\w+)\s+(\d{1,2}),?\s+(\d{4})", s, re.IGNORECASE)
    if m and m.group(1).lower() in _MONTH_MAP:
        return date(int(m.group(3)), _MONTH_MAP[m.group(1).lower()], int(m.group(2)))

    m = re.match(r"(\w+)\s+(\d{4})", s, re.IGNORECASE)
    if m and m.group(1).lower() in _MONTH_MAP:
        return date(int(m.group(2)), _MONTH_MAP[m.group(1).lower()], 1)

    return None


def extract_date_from_ocr(markdown_text: str) -> date | None:
    """Extract the paper's date from OCR'd markdown text.

    Tries multiple strategies in order of reliability:
    1. Explicit date labels (Received/Submitted/Published)
    2. arXiv ID (encodes year+month)
    3. Standalone date in the first ~2000 chars (title page area)
    4. Copyright year (least precise — only year)
    """
    if not markdown_text:
        return None

    # 1. Labeled dates (most reliable)
    for m in _DATE_LABEL_PATTERN.finditer(markdown_text[:5000]):
        parsed = _parse_date_str(m.group(1))
        if parsed:
            logger.info("Extracted paper date from label: %s", parsed)
            return parsed

    # 2. arXiv ID → year/month
    for pattern in (_ARXIV_ID_PATTERN, _ARXIV_URL_PATTERN):
        m = pattern.search(markdown_text[:5000])
        if m:
            yy, mm = int(m.group(1)), int(m.group(2))
            year = 2000 + yy if yy < 100 else yy
            if 1 <= mm <= 12:
                # Use last day of submission month as the date
                import calendar
                last_day = calendar.monthrange(year, mm)[1]
                d = date(year, mm, last_day)
                logger.info("Extracted paper date from arXiv ID: %s", d)
                return d

    # 3. Standalone date in first ~2000 chars (title page)
    for m in _STANDALONE_DATE.finditer(markdown_text[:2000]):
        parsed = _parse_date_str(m.group(1))
        if parsed:
            logger.info("Extracted paper date from title page: %s", parsed)
            return parsed

    # 4. Copyright year (least precise)
    m = _COPYRIGHT_PATTERN.search(markdown_text[:3000])
    if m:
        year = int(m.group(1))
        if 2000 <= year <= 2030:
            d = date(year, 12, 31)  # Use end of year as conservative estimate
            logger.info("Extracted paper date from copyright year: %s", d)
            return d

    return None


# ── OpenAlex fallback ────────────────────────────────────────────────────────

OPENALEX_API = "https://api.openalex.org"
OPENALEX_HEADERS = {"User-Agent": "CMUPaperReviewer/1.0 (mailto:seungone@cmu.edu)"}


def search_openalex_date(title: str) -> date | None:
    """Search OpenAlex for a paper by title and return its publication date."""
    if not title or len(title.strip()) < 10:
        return None

    try:
        resp = requests.get(
            f"{OPENALEX_API}/works",
            params={"search": title.strip(), "per-page": 3},
            headers=OPENALEX_HEADERS,
            timeout=10,
        )
        if resp.status_code != 200:
            logger.warning("OpenAlex search returned %d", resp.status_code)
            return None

        data = resp.json()
        results = data.get("results", [])
        if not results:
            return None

        # Take the first result's publication_date
        pub_date_str = results[0].get("publication_date")
        if pub_date_str:
            parsed = _parse_date_str(pub_date_str)
            if parsed:
                logger.info("Got paper date from OpenAlex: %s (title match: %s)",
                            parsed, results[0].get("title", "?")[:60])
                return parsed

        # Fallback to publication_year
        pub_year = results[0].get("publication_year")
        if pub_year:
            d = date(pub_year, 12, 31)
            logger.info("Got paper year from OpenAlex: %d", pub_year)
            return d

    except Exception:
        logger.exception("OpenAlex lookup failed")

    return None


def _extract_title_from_markdown(markdown_text: str) -> str | None:
    """Try to extract the paper title from the first lines of OCR'd markdown."""
    lines = markdown_text.strip().split("\n")
    # Skip empty lines, look for the first substantial text (likely the title)
    for line in lines[:10]:
        line = line.strip().lstrip("#").strip()
        # Title is usually 5+ words, not a date, not a header like "Abstract"
        if (len(line) > 20
                and not re.match(r"^(abstract|introduction|keywords|contents|table of)", line, re.IGNORECASE)
                and not re.match(r"^\d", line)):
            return line
    return None


def _extract_search_term_from_filename(filename: str) -> str | None:
    """Extract a usable search term from the uploaded PDF filename.

    Common patterns:
      - "2501.12345v1.pdf" (arXiv ID)
      - "Attention_Is_All_You_Need.pdf"
      - "smith2025_transformers.pdf"
    """
    if not filename:
        return None

    # Strip extension
    name = re.sub(r"\.pdf$", "", filename, flags=re.IGNORECASE).strip()
    if not name:
        return None

    # Check for arXiv ID pattern: 2501.12345 → search "arXiv 2501.12345"
    arxiv_match = re.match(r"(\d{4}\.\d{4,5})(v\d+)?$", name)
    if arxiv_match:
        return f"arXiv {arxiv_match.group(1)}"

    # Clean up common filename separators into spaces for search
    cleaned = re.sub(r"[_\-]+", " ", name)
    # Remove leading author/year patterns like "smith2025 " to get the title part
    cleaned = re.sub(r"^\w+\d{4}\s+", "", cleaned)

    if len(cleaned) > 10:
        return cleaned

    return None


def get_paper_date(markdown_text: str, filename: str | None = None) -> str | None:
    """Get the paper date as an ISO string (YYYY-MM-DD).

    Tries multiple strategies in order:
    1. OCR regex extraction (dates in the manuscript text)
    2. OpenAlex search by title extracted from OCR
    3. Filename-based search (arXiv ID or title from filename) via OpenAlex
    Returns None if the date cannot be determined.
    """
    # 1. Try OCR extraction
    d = extract_date_from_ocr(markdown_text)
    if d:
        return d.isoformat()

    # 2. Fallback: search OpenAlex by title from OCR text
    title = _extract_title_from_markdown(markdown_text)
    if title:
        d = search_openalex_date(title)
        if d:
            return d.isoformat()

    # 3. Fallback: search OpenAlex using the filename
    if filename:
        search_term = _extract_search_term_from_filename(filename)
        if search_term:
            logger.info("Trying OpenAlex with filename-derived query: %s", search_term)
            d = search_openalex_date(search_term)
            if d:
                return d.isoformat()

    logger.warning("Could not determine paper date from OCR, OpenAlex, or filename")
    return None


# ── OpenAlex date lookup for individual papers (used by Tavily MCP) ──────────

def lookup_paper_date_openalex(title: str) -> str | None:
    """Look up publication date of a specific paper via OpenAlex.

    Returns ISO date string or None.
    """
    return search_openalex_date(title)
