"""Custom Tavily MCP server that filters search results by publication date.

Usage:
    python -m backend.services.tavily_mcp --api-key <KEY> [--paper-date 2025-06-15]

When --paper-date is provided, results with published_date after that date are
excluded from the response. Results without dates are cross-checked against
OpenAlex for academic papers. When --paper-date is omitted, all results pass.
"""

import argparse
import json
import re
import sys
from datetime import datetime

import requests
from mcp.server.fastmcp import FastMCP
from tavily import TavilyClient

# ── Parse CLI args before MCP takes over stdio ──────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--api-key", required=True, help="Tavily API key")
parser.add_argument("--paper-date", default=None,
                    help="Paper date (YYYY-MM-DD). Results after this date are excluded.")
args, _unknown = parser.parse_known_args()

client = TavilyClient(api_key=args.api_key)
paper_date = None
if args.paper_date:
    try:
        paper_date = datetime.strptime(args.paper_date, "%Y-%m-%d").date()
    except ValueError:
        print(f"Warning: invalid --paper-date '{args.paper_date}', ignoring filter",
              file=sys.stderr)

mcp = FastMCP("tavily-search")

# ── OpenAlex helpers ─────────────────────────────────────────────────────────

OPENALEX_API = "https://api.openalex.org"
OPENALEX_HEADERS = {"User-Agent": "CMUPaperReviewer/1.0 (mailto:seungone@cmu.edu)"}

# Domains that indicate academic papers (worth checking OpenAlex)
_ACADEMIC_DOMAINS = re.compile(
    r"arxiv\.org|scholar\.google|semanticscholar\.org|openreview\.net|aclanthology\.org"
    r"|proceedings\.|ieee\.org|acm\.org|springer\.com|nature\.com|sciencedirect\.com"
    r"|wiley\.com|plos\.org|biorxiv\.org|medrxiv\.org|doi\.org",
    re.IGNORECASE,
)


def _looks_academic(result: dict) -> bool:
    """Check if a Tavily result looks like an academic paper."""
    url = result.get("url", "")
    return bool(_ACADEMIC_DOMAINS.search(url))


def _openalex_date(title: str):
    """Look up publication date via OpenAlex. Returns date object or None."""
    if not title or len(title.strip()) < 10:
        return None
    try:
        resp = requests.get(
            f"{OPENALEX_API}/works",
            params={"search": title.strip()[:200], "per-page": 1},
            headers=OPENALEX_HEADERS,
            timeout=8,
        )
        if resp.status_code != 200:
            return None
        results = resp.json().get("results", [])
        if not results:
            return None
        pub_str = results[0].get("publication_date")
        if pub_str:
            return datetime.strptime(pub_str, "%Y-%m-%d").date()
        pub_year = results[0].get("publication_year")
        if pub_year:
            return datetime(pub_year, 12, 31).date()
    except Exception:
        pass
    return None


# ── Date parsing & filtering ─────────────────────────────────────────────────

def _parse_date(date_str: str):
    """Parse a date string in various formats, return a date object or None."""
    from email.utils import parsedate_to_datetime

    # RFC 2822 (e.g. "Sat, 14 Mar 2026 10:25:28 GMT")
    try:
        return parsedate_to_datetime(date_str).date()
    except Exception:
        pass

    # ISO and common formats
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d", "%Y-%m-%dT%H:%M:%S.%f",
                "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S%z",
                "%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(date_str.strip(), fmt).date()
        except ValueError:
            continue
    return None


def _filter_by_date(results: list[dict]) -> list[dict]:
    """Remove results published after the paper date.

    For results without a date from Tavily, if they look like academic papers,
    cross-check with OpenAlex.
    """
    if paper_date is None:
        return results

    filtered = []
    for r in results:
        pub = r.get("published_date") or r.get("publishedDate")

        if pub:
            pub_dt = _parse_date(pub)
            if pub_dt is None:
                filtered.append(r)  # Unparseable date — keep
            elif pub_dt <= paper_date:
                filtered.append(r)
            # else: excluded
        elif _looks_academic(r):
            # No Tavily date but looks academic — try OpenAlex
            title = r.get("title", "")
            oa_date = _openalex_date(title)
            if oa_date is None:
                filtered.append(r)  # Can't verify — keep
            elif oa_date <= paper_date:
                r["_openalex_date"] = str(oa_date)
                filtered.append(r)
            # else: excluded (OpenAlex says it's too new)
        else:
            # Non-academic result without date — keep
            filtered.append(r)

    return filtered


# ── MCP tools ────────────────────────────────────────────────────────────────

@mcp.tool()
def search(query: str, max_results: int = 10, search_depth: str = "advanced",
           include_raw_content: bool = False, topic: str = "general") -> str:
    """Search the web using Tavily and return results.

    Args:
        query: The search query string.
        max_results: Maximum number of results to return (default 10).
        search_depth: "basic" or "advanced" (default "advanced").
        include_raw_content: Whether to include raw page content (default False).
        topic: Search topic - "general" or "news" (default "general").

    Returns:
        JSON string with search results.
    """
    # Request extra results to account for filtering
    fetch_count = max_results * 2 if paper_date else max_results

    response = client.search(
        query=query,
        max_results=min(fetch_count, 20),
        search_depth=search_depth,
        include_raw_content=include_raw_content,
        topic=topic,
    )

    if "results" in response:
        original_count = len(response["results"])
        response["results"] = _filter_by_date(response["results"])
        # Trim back to requested max
        response["results"] = response["results"][:max_results]
        filtered_count = original_count - len(response["results"])
        if filtered_count > 0:
            response["_filtered_count"] = filtered_count
            response["_paper_date_cutoff"] = str(paper_date)

    return json.dumps(response, default=str)


@mcp.tool()
def extract(urls: list[str]) -> str:
    """Extract content from one or more URLs using Tavily.

    Args:
        urls: List of URLs to extract content from.

    Returns:
        JSON string with extracted content.
    """
    response = client.extract(urls=urls)
    return json.dumps(response, default=str)


if __name__ == "__main__":
    mcp.run(transport="stdio")
