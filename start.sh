#!/usr/bin/env bash
set -e

# Clear previous review data on startup
rm -rf data/reviews/* data/uploads/* data/pdfs/* data/ocr/*

# Ensure data directories exist (volume mount point)
mkdir -p data/uploads data/ocr data/reviews data/pdfs

# Drop and recreate the database so schema is always up-to-date
rm -f data/reviewer.db

# Start worker in background
python -m backend.worker &
WORKER_PID=$!

# Start API server in background
uvicorn backend.main:app --host 0.0.0.0 --port 8080 &
SERVER_PID=$!

# Trap signals for clean shutdown
cleanup() {
    echo "Shutting down..."
    kill "$WORKER_PID" "$SERVER_PID" 2>/dev/null || true
    wait "$WORKER_PID" "$SERVER_PID" 2>/dev/null || true
    exit 0
}
trap cleanup SIGTERM SIGINT

# Wait for either process to exit
wait -n "$WORKER_PID" "$SERVER_PID"
cleanup
