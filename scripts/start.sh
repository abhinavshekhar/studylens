#!/usr/bin/env bash
set -euo pipefail
cd /workspace

export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

if curl -sf -o /dev/null --max-time 2 http://127.0.0.1:8501/; then
  echo "StudyLens is already running on port 8501"
  exit 0
fi

exec .venv/bin/streamlit run app.py \
  --server.port 8501 \
  --server.address 0.0.0.0 \
  --server.headless true
