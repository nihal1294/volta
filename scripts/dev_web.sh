#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PORT="${WEB_PORT:-${VOLTA_WEB_PORT:-4173}}"

cd "$ROOT/apps/web"
exec npm run dev -- --host 127.0.0.1 --port "$PORT" --strictPort
