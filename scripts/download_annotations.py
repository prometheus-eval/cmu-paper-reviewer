#!/usr/bin/env python3
"""Download all annotations from the CMU Paper Reviewer API to a local JSON file.

Usage:
    python scripts/download_annotations.py                            # reads key from .env
    python scripts/download_annotations.py --key YOUR_ADMIN_KEY       # explicit key
    python scripts/download_annotations.py --api http://localhost:8000 # local server
    python scripts/download_annotations.py -o my_annotations.json     # custom output
"""

import argparse
import json
import sys
import urllib.request
import urllib.error

DEFAULT_API = "https://cmu-paper-reviewer.fly.dev"


def load_admin_key_from_env():
    """Try to read ADMIN_API_KEY from .env file."""
    from pathlib import Path

    env_file = Path(__file__).resolve().parent.parent / ".env"
    if not env_file.exists():
        return None
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line.startswith("ADMIN_API_KEY="):
            return line.split("=", 1)[1].strip()
    return None


def main():
    parser = argparse.ArgumentParser(description="Download annotations from CMU Paper Reviewer")
    parser.add_argument("--api", default=DEFAULT_API, help=f"API base URL (default: {DEFAULT_API})")
    parser.add_argument("--key", default=None, help="Admin API key (default: reads from .env)")
    parser.add_argument("-o", "--output", default="annotations.json", help="Output file path (default: annotations.json)")
    args = parser.parse_args()

    admin_key = args.key or load_admin_key_from_env()
    if not admin_key:
        print("Error: No admin API key provided.", file=sys.stderr)
        print("  Pass --key YOUR_KEY or set ADMIN_API_KEY in .env", file=sys.stderr)
        sys.exit(1)

    url = f"{args.api.rstrip('/')}/api/annotations/export"
    print(f"Fetching annotations from {url} ...")

    try:
        req = urllib.request.Request(url, headers={"X-Admin-Key": admin_key})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        if e.code == 401:
            print("Error: Invalid admin API key.", file=sys.stderr)
        else:
            print(f"HTTP error {e.code}: {body}", file=sys.stderr)
        sys.exit(1)
    except urllib.error.URLError as e:
        print(f"Connection error: {e.reason}", file=sys.stderr)
        sys.exit(1)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Downloaded {len(data)} annotation(s) to {args.output}")


if __name__ == "__main__":
    main()
