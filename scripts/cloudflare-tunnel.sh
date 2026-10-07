#!/usr/bin/env bash
set -euo pipefail

TARGET_URL="${STUDYLENS_TUNNEL_TARGET:-http://127.0.0.1:8501}"
LOG_FILE="${STUDYLENS_TUNNEL_LOG:-/tmp/cloudflared-studylens.log}"

if ! command -v cloudflared >/dev/null 2>&1; then
  echo "cloudflared is not installed. Install from https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/"
  exit 1
fi

if curl -sf -o /dev/null --max-time 2 "${TARGET_URL}/"; then
  echo "Tunnel target is up: ${TARGET_URL}"
else
  echo "Warning: ${TARGET_URL} is not responding yet. Tunnel will still start."
fi

if pgrep -f "cloudflared tunnel --url ${TARGET_URL}" >/dev/null 2>&1; then
  existing_url="$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "${LOG_FILE}" 2>/dev/null | tail -1 || true)"
  if [ -n "${existing_url}" ]; then
    echo "Cloudflare tunnel already running: ${existing_url}"
    exit 0
  fi
fi

exec cloudflared tunnel --url "${TARGET_URL}" 2>&1 | tee -a "${LOG_FILE}"
