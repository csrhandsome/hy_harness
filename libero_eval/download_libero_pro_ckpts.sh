#!/usr/bin/env bash
# Download VLA-Adapter LIBERO *-Pro checkpoints (one suite ↔ one ckpt).
#
# Usage:
#   bash vla-scripts/download_libero_pro_ckpts.sh
#   FORCE=1 bash vla-scripts/download_libero_pro_ckpts.sh
#   HF_ENDPOINT=https://huggingface.co bash vla-scripts/download_libero_pro_ckpts.sh
#   ONLY=libero_object bash vla-scripts/download_libero_pro_ckpts.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/outputs}"
MAX_WORKERS="${MAX_WORKERS:-4}"
FORCE="${FORCE:-0}"
ONLY="${ONLY:-}"

# Prefer mirror; override with HF_ENDPOINT=https://huggingface.co if needed.
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-300}"

# suite → HF repo → local dir name
CKPTS=(
  "libero_spatial|VLA-Adapter/LIBERO-Spatial-Pro|LIBERO-Spatial-Pro"
  "libero_object|VLA-Adapter/LIBERO-Object-Pro|LIBERO-Object-Pro"
  "libero_goal|VLA-Adapter/LIBERO-Goal-Pro|LIBERO-Goal-Pro"
  "libero_10|VLA-Adapter/LIBERO-Long-Pro|LIBERO-Long-Pro"
)

ckpt_ready() {
  local dir="$1"
  [[ -d "${dir}" ]] && [[ -f "${dir}/config.json" ]] && \
    [[ "$(find "${dir}" -maxdepth 1 -name '*.safetensors' -o -name '*.bin' -o -name '*.pt' 2>/dev/null | wc -l)" -ge 1 ]]
}

mkdir -p "${OUT_DIR}"

for entry in "${CKPTS[@]}"; do
  IFS='|' read -r suite repo local_name <<<"${entry}"
  if [[ -n "${ONLY}" && "${ONLY}" != "${suite}" && "${ONLY}" != "${local_name}" ]]; then
    continue
  fi

  dest="${OUT_DIR}/${local_name}"
  if ckpt_ready "${dest}" && [[ "${FORCE}" != "1" ]]; then
    echo "Skip (already present): ${dest}"
    du -sh "${dest}"
    continue
  fi

  echo "Downloading ${repo} → ${dest} via ${HF_ENDPOINT} ..."
  hf download "${repo}" \
    --local-dir "${dest}" \
    --max-workers "${MAX_WORKERS}"

  if ! ckpt_ready "${dest}"; then
    echo "ERROR: download incomplete for ${dest}"
    exit 1
  fi
  echo "Done: ${dest}"
  du -sh "${dest}"
done

echo "All requested Pro checkpoints ready under: ${OUT_DIR}"
