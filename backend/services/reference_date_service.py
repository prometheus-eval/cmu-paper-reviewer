"""Ground reference dates and tag review citations as [BEFORE]/[AFTER].

This runs as a deterministic post-pass after the review is generated, replacing
the review agent's unreliable from-memory guesses. The logic mirrors the
intended user flow:

1. Determine whether the submitted manuscript already exists online and, if so,
   its publication date. If it cannot be found online it is treated as an
   unreleased manuscript and *every* reference is tagged [BEFORE].
2. For each reference in the citation list, look up its publication date
   individually: arXiv id in the link (exact) -> Tavily web search (if a key is
   configured) -> OpenAlex by title.
3. Compare each reference's date against the manuscript date to decide the tag.

Dates are kept as PartialDate (year / optional month / optional day) so that a
year-only citation is compared against a precise manuscript date at the coarsest
shared granularity, avoiding spurious same-year flips.
"""

import calendar
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from email.utils import parsedate_to_datetime

import requests

from backend.services.paper_date_service import (
    OPENALEX_API,
    OPENALEX_HEADERS,
    _MONTH_MAP,
    _extract_title_from_markdown,
)

logger = logging.getLogger(__name__)


# ── Partial date (precision-aware) ───────────────────────────────────────────

@dataclass
class PartialDate:
    year: int
    month: int | None = None
    day: int | None = None

    def __str__(self) -> str:
        if self.month and self.day:
            return f"{self.year:04d}-{self.month:02d}-{self.day:02d}"
        if self.month:
            return f"{self.year:04d}-{self.month:02d}"
        return f"{self.year:04d}"


def _is_before_or_equal(ref: PartialDate, manu: PartialDate) -> bool:
    """True if `ref` was published on or before `manu`, compared at the coarsest
    granularity both share. Ties / unresolvable cases count as BEFORE so the
    authors are never penalized for a borderline date."""
    if ref.year != manu.year:
        return ref.year < manu.year
    if ref.month and manu.month and ref.month != manu.month:
        return ref.month < manu.month
    if ref.month and manu.month and ref.month == manu.month and ref.day and manu.day:
        return ref.day <= manu.day
    return True  # same year (or missing month/day) → ambiguous → BEFORE


# ── Date string parsing ──────────────────────────────────────────────────────

_ISO_DATE = re.compile(r"(\d{4})-(\d{1,2})-(\d{1,2})")
_YEAR_ONLY = re.compile(r"^\s*(\d{4})\s*$")
_BARE_YEAR = re.compile(r"\b(19|20)\d{2}\b")


def _partial_from_str(s: str) -> PartialDate | None:
    """Parse an academic date string (ISO, 'Month YYYY', '15 January 2025', …)
    into a PartialDate, preserving the available precision."""
    if not s:
        return None
    s = s.strip().rstrip(".")

    m = _ISO_DATE.match(s)
    if m:
        return PartialDate(int(m.group(1)), int(m.group(2)), int(m.group(3)))

    m = _YEAR_ONLY.match(s)
    if m:
        return PartialDate(int(m.group(1)))

    # "15 January 2025"
    m = re.match(r"(\d{1,2})\s+(\w+)\s+(\d{4})", s, re.IGNORECASE)
    if m and m.group(2).lower() in _MONTH_MAP:
        return PartialDate(int(m.group(3)), _MONTH_MAP[m.group(2).lower()], int(m.group(1)))

    # "January 15, 2025"
    m = re.match(r"(\w+)\s+(\d{1,2}),?\s+(\d{4})", s, re.IGNORECASE)
    if m and m.group(1).lower() in _MONTH_MAP:
        return PartialDate(int(m.group(3)), _MONTH_MAP[m.group(1).lower()], int(m.group(2)))

    # "January 2025"
    m = re.match(r"(\w+)\s+(\d{4})", s, re.IGNORECASE)
    if m and m.group(1).lower() in _MONTH_MAP:
        return PartialDate(int(m.group(2)), _MONTH_MAP[m.group(1).lower()])

    return None


def _partial_from_web_date(s: str) -> PartialDate | None:
    """Parse a web `published_date` (RFC 2822 or ISO datetime) into a PartialDate."""
    if not s:
        return None
    try:
        dt = parsedate_to_datetime(s)
        if dt:
            return PartialDate(dt.year, dt.month, dt.day)
    except Exception:
        pass
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(s.strip(), fmt)
            return PartialDate(dt.year, dt.month, dt.day)
        except ValueError:
            continue
    return _partial_from_str(s)


# ── arXiv id (exact, no network) ─────────────────────────────────────────────

_ARXIV_IN_URL = re.compile(r"arxiv\.org/(?:abs|pdf)/(\d{2})(\d{2})\.\d{4,5}", re.IGNORECASE)
_ARXIV_ID = re.compile(r"\barXiv:\s*(\d{2})(\d{2})\.\d{4,5}", re.IGNORECASE)


def _arxiv_partial(text: str | None) -> PartialDate | None:
    """Decode the YYMM in an arXiv id/URL into a PartialDate (year + month)."""
    if not text:
        return None
    for pat in (_ARXIV_IN_URL, _ARXIV_ID):
        m = pat.search(text)
        if m:
            yy, mm = int(m.group(1)), int(m.group(2))
            if 1 <= mm <= 12:
                return PartialDate(2000 + yy, mm)
    return None


# ── Title similarity (to match the right search result) ──────────────────────

_STOPWORDS = {
    "a", "an", "the", "of", "for", "and", "or", "to", "in", "on", "with", "via",
    "is", "are", "be", "by", "from", "using", "use", "we", "our", "this", "that",
    "towards", "toward", "into", "as", "at",
}


def _title_words(title: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", (title or "").lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 1}


def _title_similar(a: str, b: str, threshold: float = 0.5) -> bool:
    wa, wb = _title_words(a), _title_words(b)
    if not wa or not wb:
        return False
    inter = len(wa & wb)
    union = len(wa | wb)
    return union > 0 and inter / union >= threshold


def _same_paper_url(a: str, b: str) -> bool:
    """Heuristic: do two URLs point at the same work? (arXiv id match, else exact path)."""
    ax, bx = _arxiv_partial(a), _arxiv_partial(b)
    aid = _ARXIV_IN_URL.search(a or "")
    bid = _ARXIV_IN_URL.search(b or "")
    if aid and bid:
        return aid.group(0).lower() == bid.group(0).lower()

    def norm(u: str) -> str:
        u = re.sub(r"^https?://", "", (u or "").lower())
        u = re.sub(r"^www\.", "", u)
        return u.rstrip("/")

    return bool(a) and bool(b) and norm(a) == norm(b)


# ── Source lookups ───────────────────────────────────────────────────────────

_DOI = re.compile(r"10\.\d{4,9}/[^\s)\]>]+", re.IGNORECASE)


def _openalex_by_doi(url: str | None) -> PartialDate | None:
    """Exact lookup: resolve a DOI found in the citation link via OpenAlex."""
    if not url:
        return None
    m = _DOI.search(url)
    if not m:
        return None
    doi = m.group(0).rstrip(".")
    try:
        resp = requests.get(
            f"{OPENALEX_API}/works/doi:{doi}",
            headers=OPENALEX_HEADERS,
            timeout=10,
        )
        if resp.status_code != 200:
            return None
        r = resp.json()
    except Exception:
        logger.warning("OpenAlex DOI lookup failed for %s", doi[:80])
        return None
    pub = r.get("publication_date")
    if pub:
        return _partial_from_str(pub)
    if r.get("publication_year"):
        return PartialDate(int(r["publication_year"]))
    return None


def _openalex_partial(title: str) -> PartialDate | None:
    if not title or len(title.strip()) < 8:
        return None
    try:
        resp = requests.get(
            f"{OPENALEX_API}/works",
            params={"search": title.strip()[:300], "per-page": 10},
            headers=OPENALEX_HEADERS,
            timeout=10,
        )
        if resp.status_code != 200:
            return None
        results = resp.json().get("results", [])
    except Exception:
        logger.exception("OpenAlex lookup failed for %r", title[:60])
        return None

    def _result_date(r: dict) -> PartialDate | None:
        pub = r.get("publication_date")
        if pub:
            return _partial_from_str(pub)
        if r.get("publication_year"):
            return PartialDate(int(r["publication_year"]))
        return None

    # Collect dates from results whose title actually matches. A title search can
    # return reprints, surveys, or citing works dated later than the original, so
    # we take the *earliest* matching date — this is the original publication and
    # the most charitable to the authors. Unmatched results are ignored entirely
    # (returning their date would be worse than the citation-year fallback).
    candidates: list[PartialDate] = []
    for r in results:
        cand_title = r.get("display_name") or r.get("title") or ""
        if _title_similar(title, cand_title):
            d = _result_date(r)
            if d:
                candidates.append(d)
    if not candidates:
        return None
    return min(candidates, key=lambda p: (p.year, p.month or 1, p.day or 1))


def _tavily_partial(
    title: str, ref_url: str | None, client, require_url_match: bool = False
) -> PartialDate | None:
    """Search Tavily and return a result's published_date. With
    require_url_match, only a result whose URL matches the citation link counts
    (high confidence); otherwise a title-similar result is also accepted."""
    if client is None or not title:
        return None
    try:
        resp = client.search(query=title[:380], max_results=5, search_depth="basic")
    except Exception:
        logger.warning("Tavily lookup failed for %r", title[:60])
        return None

    results = resp.get("results", []) if isinstance(resp, dict) else []
    title_fallback: PartialDate | None = None
    for r in results:
        pub = r.get("published_date") or r.get("publishedDate")
        if not pub:
            continue
        pd = _partial_from_web_date(pub)
        if pd is None:
            continue
        if ref_url and r.get("url") and _same_paper_url(ref_url, r["url"]):
            return pd
        if title_fallback is None and _title_similar(title, r.get("title", "")):
            title_fallback = pd
    return None if require_url_match else title_fallback


def lookup_reference_date_exact(
    title: str, url: str | None, client
) -> tuple[PartialDate | None, str | None]:
    """High-confidence, link-based lookups only. Returns (PartialDate, source)."""
    # 1. arXiv id in the link or text — exact, no network call.
    pd = _arxiv_partial(url) or _arxiv_partial(title)
    if pd:
        return pd, "arxiv"
    # 2. DOI in the link — exact OpenAlex resolution.
    pd = _openalex_by_doi(url)
    if pd:
        return pd, "openalex-doi"
    # 3. Tavily result whose URL matches the citation link.
    pd = _tavily_partial(title, url, client, require_url_match=True)
    if pd:
        return pd, "tavily-url"
    return None, None


def lookup_reference_date_weak(
    title: str, url: str | None, client
) -> tuple[PartialDate | None, str | None]:
    """Title-only lookups (noisier). Used only when no exact signal or printed
    year is available. Returns (PartialDate, source)."""
    pd = _tavily_partial(title, url, client, require_url_match=False)
    if pd:
        return pd, "tavily-title"
    pd = _openalex_partial(title)
    if pd:
        return pd, "openalex-title"
    return None, None


# ── Manuscript existence + date ──────────────────────────────────────────────

def determine_manuscript_date(
    ocr_text: str, filename: str | None, client
) -> tuple[PartialDate | None, bool]:
    """Determine whether the manuscript exists online and its publication date.

    Returns (manuscript_date, exists). When the manuscript cannot be found
    online it is treated as unreleased: exists=False (callers tag every
    reference [BEFORE]).
    """
    # An arXiv id in the manuscript itself means it is already public.
    arxiv = _arxiv_partial(ocr_text[:4000]) or _arxiv_partial(filename or "")
    if arxiv:
        logger.info("Manuscript found via arXiv id: %s", arxiv)
        return arxiv, True

    title = _extract_title_from_markdown(ocr_text or "")
    if not title:
        logger.info("No manuscript title extracted; treating as unreleased.")
        return None, False

    # Tavily first (if available): require a real title match to count as "exists".
    if client is not None:
        try:
            resp = client.search(query=title[:380], max_results=5, search_depth="basic")
            for r in (resp.get("results", []) if isinstance(resp, dict) else []):
                if _title_similar(title, r.get("title", "")):
                    pub = r.get("published_date") or r.get("publishedDate")
                    pd = _partial_from_web_date(pub) if pub else None
                    logger.info("Manuscript matched on Tavily (date=%s): %s", pd, title[:60])
                    return pd, True
        except Exception:
            logger.warning("Tavily manuscript lookup failed for %r", title[:60])

    # OpenAlex fallback (also requires a title match inside _openalex_partial).
    pd = _openalex_partial(title)
    if pd:
        logger.info("Manuscript matched on OpenAlex (date=%s): %s", pd, title[:60])
        return pd, True

    logger.info("Manuscript not found online; treating as unreleased: %s", title[:60])
    return None, False


# ── Citation-list tagging ────────────────────────────────────────────────────

_CITATION_LINE = re.compile(r"^\s*\[(\d+)\]\s+(.+)$")
_CITATION_HEADER = re.compile(r"^#{1,6}\s*citation list", re.IGNORECASE)
_SECTION_HEADER = re.compile(r"^#{1,6}\s+\S")
_TAG_SUFFIX = re.compile(r"\s*\[(?:BEFORE|AFTER)\]\s*$", re.IGNORECASE)
_LINK_URL = re.compile(r"\[[^\]]*\]\((https?://[^)\s]+)\)")
_QUOTED_TITLE = re.compile(r'["“”‘’\']([^"“”‘’\']{6,})["“”‘’\']')


def _extract_title_and_url(body: str) -> tuple[str, str | None]:
    """Pull a search title and link URL out of a citation line body."""
    url_m = _LINK_URL.search(body)
    url = url_m.group(1) if url_m else None

    q = _QUOTED_TITLE.search(body)
    if q:
        return q.group(1).strip().rstrip(",;.").strip(), url

    # No quoted title — take the text before the first comma/period, minus links.
    cleaned = _LINK_URL.sub("", body)
    cleaned = re.sub(r"^\s*[A-Z][\w.\-]+ et al\.?,?\s*", "", cleaned)  # drop "Smith et al.,"
    title = re.split(r"[,.]", cleaned, maxsplit=1)[0].strip()
    return title, url


def _decide_tag(
    body: str, manu_date: PartialDate | None, manu_exists: bool, client
) -> str | None:
    """Return '[BEFORE]', '[AFTER]', or None (leave untagged when unknown)."""
    if not manu_exists:
        return "[BEFORE]"  # unreleased manuscript → everything predates it

    title, url = _extract_title_and_url(body)

    # Precedence: exact link-based lookups (arXiv/DOI/URL-matched) are most
    # reliable; below them the year the author printed in the citation beats
    # noisy title-only web/OpenAlex matches (which can return the wrong record).
    ref_date, source = lookup_reference_date_exact(title, url, client)

    if ref_date is None:
        ym = _BARE_YEAR.search(body)
        if ym:
            ref_date, source = PartialDate(int(ym.group(0))), "citation-year"

    if ref_date is None:
        ref_date, source = lookup_reference_date_weak(title, url, client)

    if ref_date is None or manu_date is None:
        logger.info("Citation date unresolved; leaving untagged: %r", title[:60])
        return None

    tag = "[BEFORE]" if _is_before_or_equal(ref_date, manu_date) else "[AFTER]"
    logger.info("Citation %r: ref=%s (%s) vs manuscript=%s -> %s",
                title[:50], ref_date, source, manu_date, tag)
    return tag


def tag_review_citations(
    review_md: str, manu_date: PartialDate | None, manu_exists: bool, client
) -> str:
    """Rewrite [BEFORE]/[AFTER] tags on every line of the review's citation list."""
    lines = review_md.split("\n")
    out: list[str] = []
    in_list = False

    for line in lines:
        if _CITATION_HEADER.match(line.strip()):
            in_list = True
            out.append(line)
            continue
        # A new section header (that isn't a citation line) ends the list.
        if in_list and _SECTION_HEADER.match(line) and not _CITATION_LINE.match(line):
            in_list = False

        m = _CITATION_LINE.match(line) if in_list else None
        if not m:
            out.append(line)
            continue

        base = _TAG_SUFFIX.sub("", line)  # strip any existing/agent-added tag
        tag = _decide_tag(m.group(2), manu_date, manu_exists, client)
        out.append(f"{base} {tag}" if tag else base)

    return "\n".join(out)


# ── Top-level entry point (called by the worker) ─────────────────────────────

def tag_review_dates(key: str, tavily_api_key: str | None, filename: str | None = None) -> None:
    """Post-process a generated review: determine the manuscript date from its
    OCR text (and online presence), then tag each citation [BEFORE]/[AFTER].
    Rewrites the review markdown in place. Best-effort; never raises."""
    from backend.services.storage_service import preprint_md_path, review_md_path

    review_path = review_md_path(key)
    if not review_path.exists():
        logger.info("[%s] No review markdown to tag.", key)
        return

    md_path = preprint_md_path(key)
    ocr_text = md_path.read_text(encoding="utf-8") if md_path.exists() else ""

    client = None
    if tavily_api_key:
        try:
            from tavily import TavilyClient
            client = TavilyClient(api_key=tavily_api_key)
        except Exception:
            logger.warning("[%s] Could not init Tavily client; using OpenAlex only.", key)

    manu_date, exists = determine_manuscript_date(ocr_text, filename, client)
    logger.info("[%s] Manuscript exists=%s date=%s", key, exists, manu_date)

    review_md = review_path.read_text(encoding="utf-8")
    tagged = tag_review_citations(review_md, manu_date, exists, client)
    if tagged != review_md:
        review_path.write_text(tagged, encoding="utf-8")
        logger.info("[%s] Citation date tags updated.", key)
