#!/usr/bin/env bash
# Download RPent text memory + seed-0 recipes for the Harness via hf-mirror.
#
# Usage:
#   bash sde_harness/scripts/download_rpent_memory.sh
#   HF_ENDPOINT=https://huggingface.co bash sde_harness/scripts/download_rpent_memory.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEST="${DEST:-${ROOT_DIR}/sde_harness/resources}"
MAX_WORKERS="${MAX_WORKERS:-4}"
if [[ "${MAX_WORKERS}" -gt 4 ]]; then
  echo "WARNING: MAX_WORKERS=${MAX_WORKERS} > 4 tends to 429 on hf-mirror; clamping to 4"
  MAX_WORKERS=4
fi

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"

mkdir -p "${DEST}"
echo "Downloading RLinf/RPent-memory (libero/**) → ${DEST} via ${HF_ENDPOINT} ..."
hf download RLinf/RPent-memory \
  --repo-type dataset \
  --local-dir "${DEST}" \
  --include "libero/**" \
  --max-workers "${MAX_WORKERS}"

if [[ ! -f "${DEST}/libero/memory/MEMORY.md" ]]; then
  echo "ERROR: MEMORY.md missing under ${DEST}/libero/memory/"
  exit 1
fi
echo "Done:"
du -sh "${DEST}/libero" "${DEST}/libero/memory"
