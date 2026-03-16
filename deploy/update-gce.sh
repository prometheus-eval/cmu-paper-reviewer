#!/usr/bin/env bash
set -euo pipefail

cd ~/cmu-paper-reviewer
git pull
cd deploy
docker compose up -d --build
