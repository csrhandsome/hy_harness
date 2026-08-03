#!/usr/bin/env bash
# Stop the background proxy started by start_proxy.sh.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${LOG_DIR:-/tmp/codex_proxy}"
PID_FILE="${LOG_DIR}/codex_proxy.pid"

if [[ ! -f "${PID_FILE}" ]]; then
  echo "no pid file at ${PID_FILE}; nothing to stop"
  exit 0
fi

pid="$(cat "${PID_FILE}")"
if [[ -z "${pid}" ]]; then
  echo "empty pid file; removing"
  rm -f "${PID_FILE}"
  exit 0
fi

if ! kill -0 "${pid}" 2>/dev/null; then
  echo "pid ${pid} no longer running; removing pid file"
  rm -f "${PID_FILE}"
  exit 0
fi

kill "${pid}"
for _ in $(seq 1 20); do
  if ! kill -0 "${pid}" 2>/dev/null; then
    break
  fi
  sleep 0.25
done
if kill -0 "${pid}" 2>/dev/null; then
  kill -9 "${pid}" 2>/dev/null || true
fi
rm -f "${PID_FILE}"
echo "stopped codex_proxy (pid=${pid})"
