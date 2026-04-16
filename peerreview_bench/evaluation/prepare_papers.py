#!/usr/bin/env python3
"""
Wrapper around download_papers.py that sets up the evaluation paper dirs.

If papers are already downloaded, this is a no-op (unless --force is passed).

Usage:
    python3 prepare_papers.py                    # default: ../papers/
    python3 prepare_papers.py --limit 5          # first 5 papers
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_BENCH_DIR = Path(__file__).resolve().parent.parent


def prepare_papers(
    output_dir: Path = _BENCH_DIR / "papers",
    limit: int | None = None,
    skip_existing: bool = True,
    force: bool = False,
):
    """Download papers via download_papers.py."""
    script = _BENCH_DIR / "download_papers.py"
    cmd = [sys.executable, str(script), "--output-dir", str(output_dir)]
    if limit:
        cmd += ["--limit", str(limit)]
    if skip_existing and not force:
        cmd += ["--skip-existing"]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Prepare papers for evaluation")
    parser.add_argument("--output-dir", type=Path, default=_BENCH_DIR / "papers")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="Re-download existing papers")
    args = parser.parse_args()
    prepare_papers(args.output_dir, args.limit, not args.force, args.force)


if __name__ == "__main__":
    main()
