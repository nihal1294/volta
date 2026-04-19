#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WEB_PORT="${WEB_PORT:-${VOLTA_WEB_PORT:-4173}}"
API_PID=""
WEB_PID=""

cleanup() {
  local exit_code=$?
  if [[ -n "${API_PID}" ]] && kill -0 "${API_PID}" 2>/dev/null; then
    kill "${API_PID}" 2>/dev/null || true
  fi
  if [[ -n "${WEB_PID}" ]] && kill -0 "${WEB_PID}" 2>/dev/null; then
    kill "${WEB_PID}" 2>/dev/null || true
  fi
  wait "${API_PID:-}" 2>/dev/null || true
  wait "${WEB_PID:-}" 2>/dev/null || true
  exit "${exit_code}"
}

trap cleanup EXIT INT TERM

"$ROOT/scripts/bootstrap_api.sh"
"$ROOT/scripts/bootstrap_web.sh"
python3 "$ROOT/scripts/doctor.py"

(
  cd "$ROOT"
  exec uv run --project apps/api volta-api
) &
API_PID=$!

(
  cd "$ROOT/apps/web"
  exec npm run dev -- --host 127.0.0.1 --port "$WEB_PORT" --strictPort
) &
WEB_PID=$!

echo "Volta backend: http://127.0.0.1:8765"
echo "Volta frontend: http://127.0.0.1:${WEB_PORT}"

while kill -0 "$API_PID" 2>/dev/null && kill -0 "$WEB_PID" 2>/dev/null; do
  sleep 1
done
