#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
python scripts/download_annotations.py "$@"
