#!/usr/bin/env bash
# Download & install LIBERO-plus assets via hf-mirror (4 workers, no rate-limit).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIBERO_PLUS_PKG="${ROOT_DIR}/third_party/LIBERO-plus/libero/libero"
ASSETS_DIR="${LIBERO_PLUS_PKG}/assets"
LOCAL_DIR="${LOCAL_DIR:-/tmp/libero_plus_hf}"
MAX_WORKERS="${MAX_WORKERS:-4}"
FORCE="${FORCE:-0}"

# Prefer mirror; override with HF_ENDPOINT=https://huggingface.co if needed.
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-300}"

if [[ -d "${ASSETS_DIR}/new_objects" && "${FORCE}" != "1" ]]; then
  echo "LIBERO-plus assets already present at: ${ASSETS_DIR}"
  du -sh "${ASSETS_DIR}"
  exit 0
fi

mkdir -p "${LOCAL_DIR}" "${LIBERO_PLUS_PKG}"

echo "Downloading assets.zip via ${HF_ENDPOINT} (max-workers=${MAX_WORKERS}) ..."
hf download Sylvest/LIBERO-plus assets.zip \
  --repo-type dataset \
  --local-dir "${LOCAL_DIR}" \
  --max-workers "${MAX_WORKERS}"

echo "Unzipping into ${LIBERO_PLUS_PKG} ..."
unzip -q -o "${LOCAL_DIR}/assets.zip" -d "${LIBERO_PLUS_PKG}"

# HF zip nests assets under inspire/.../LIBERO-plus-0/assets
if [[ ! -d "${ASSETS_DIR}" || ! -d "${ASSETS_DIR}/new_objects" ]]; then
  NESTED="$(find "${LIBERO_PLUS_PKG}" -type d -path '*/LIBERO-plus-0/assets' 2>/dev/null | head -1 || true)"
  if [[ -n "${NESTED}" ]]; then
    rm -rf "${ASSETS_DIR}"
    mv "${NESTED}" "${ASSETS_DIR}"
    rm -rf "${LIBERO_PLUS_PKG}/inspire"
  fi
fi

if [[ ! -d "${ASSETS_DIR}/new_objects" ]]; then
  echo "ERROR: assets install failed; expected ${ASSETS_DIR}/new_objects"
  exit 1
fi

echo "Done. Assets ready at: ${ASSETS_DIR}"
du -sh "${ASSETS_DIR}"
