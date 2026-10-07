#!/usr/bin/env bash
set -euo pipefail
cd /workspace

export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
STREAMLIT_PID=""

if curl -sf -o /dev/null --max-time 2 http://127.0.0.1:8501/; then
  echo "StudyLens is already running on port 8501"
else
  .venv/bin/streamlit run app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true &
  STREAMLIT_PID=$!
  for _ in $(seq 1 30); do
    if curl -sf -o /dev/null --max-time 2 http://127.0.0.1:8501/; then
      echo "StudyLens started on port 8501 (pid ${STREAMLIT_PID})"
      break
    fi
    sleep 1
  done
  if ! curl -sf -o /dev/null --max-time 2 http://127.0.0.1:8501/; then
    echo "StudyLens failed to start on port 8501"
    kill "${STREAMLIT_PID}" 2>/dev/null || true
    exit 1
  fi
fi

if [ "${STUDYLENS_ENABLE_CLOUDFLARE_TUNNEL:-true}" = "true" ] && [ -x scripts/cloudflare-tunnel.sh ]; then
  echo "Starting Cloudflare tunnel (set STUDYLENS_ENABLE_CLOUDFLARE_TUNNEL=false to skip)..."
  exec bash scripts/cloudflare-tunnel.sh
fi

if [ -n "${STREAMLIT_PID}" ]; then
  wait "${STREAMLIT_PID}"
fi
