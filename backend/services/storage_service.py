"""File path management for uploads and reviews under data/."""

import json
import os
import re
from pathlib import Path

from backend.config import settings


def _data_dir() -> Path:
    return Path(settings.data_dir)


def uploads_dir() -> Path:
    d = _data_dir() / "uploads"
    d.mkdir(parents=True, exist_ok=True)
    return d


def review_dir(key: str) -> Path:
    d = _data_dir() / "reviews" / key
    d.mkdir(parents=True, exist_ok=True)
    return d


def preprint_dir(key: str) -> Path:
    d = review_dir(key) / "preprint"
    d.mkdir(parents=True, exist_ok=True)
    return d


def review_output_dir(key: str) -> Path:
    d = review_dir(key) / "review"
    d.mkdir(parents=True, exist_ok=True)
    return d


def upload_path(key: str, filename: str) -> Path:
    return uploads_dir() / f"{key}_{filename}"


def preprint_md_path(key: str) -> Path:
    return preprint_dir(key) / "preprint.md"


def images_dir(key: str) -> Path:
    d = preprint_dir(key) / "images"
    d.mkdir(parents=True, exist_ok=True)
    return d


def images_list_path(key: str) -> Path:
    return preprint_dir(key) / "images_list.json"


def review_md_path(key: str) -> Path:
    return review_output_dir(key) / "review.md"


def review_pdf_path(key: str) -> Path:
    return review_output_dir(key) / "review.pdf"


def supplementary_dir(key: str) -> Path:
    d = preprint_dir(key) / "supplementary"
    d.mkdir(parents=True, exist_ok=True)
    return d


def code_dir(key: str) -> Path:
    d = preprint_dir(key) / "code"
    d.mkdir(parents=True, exist_ok=True)
    return d


def verification_code_dir(key: str) -> Path:
    """Return the directory where the reviewer agent stores verification code."""
    output = review_output_dir(key)
    # The agent writes to verification_code_{model_short}/ — find any matching dir
    for child in output.iterdir():
        if child.is_dir() and child.name.startswith("verification_code"):
            return child
    return output / "verification_code"


def list_verification_code_files(key: str) -> list[dict]:
    """Return list of verification code files with names and relative paths."""
    vdir = verification_code_dir(key)
    if not vdir.exists():
        return []
    files = []
    for f in sorted(vdir.rglob("*")):
        if f.is_file():
            files.append({
                "name": str(f.relative_to(vdir)),
                "size": f.stat().st_size,
                "path": str(f),
            })
    return files


def annotations_path(key: str) -> Path:
    return review_output_dir(key) / "annotations.json"


def get_review_markdown(key: str) -> str | None:
    path = review_md_path(key)
    if path.exists():
        return path.read_text(encoding="utf-8")
    return None


EVENT_FILE_RE = re.compile(r"^event-(\d{5})-.*\.json$")


def find_trajectory_events(key: str) -> list[dict]:
    """Find and read all event files from the trajectory directory.

    Scans review_output_dir(key) for *_trajectory/ dirs, then looks
    for an events/ subdirectory (possibly nested under a conversation_id).
    Returns parsed event dicts sorted by index.
    """
    output_dir = review_output_dir(key)
    if not output_dir.exists():
        return []

    # Find trajectory directories
    traj_dirs = list(output_dir.glob("*_trajectory"))
    if not traj_dirs:
        return []

    # Search for events/ directory (may be directly under trajectory or under a conversation_id subdir)
    events_dir = None
    for traj_dir in traj_dirs:
        candidate = traj_dir / "events"
        if candidate.is_dir():
            events_dir = candidate
            break
        # Check for {conversation_id}/events/ pattern
        for sub in traj_dir.iterdir():
            if sub.is_dir():
                candidate = sub / "events"
                if candidate.is_dir():
                    events_dir = candidate
                    break
        if events_dir:
            break

    if not events_dir:
        return []

    # Read and sort event files
    events = []
    for f in events_dir.iterdir():
        m = EVENT_FILE_RE.match(f.name)
        if not m:
            continue
        idx = int(m.group(1))
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            data["_idx"] = idx
            events.append(data)
        except (json.JSONDecodeError, OSError):
            continue

    events.sort(key=lambda e: e["_idx"])
    return events
