#!/usr/bin/env bash
# Download & install LIBERO-Pro bddl/init suites via hf-mirror (4 workers).
#
# Usage:
#   bash vla-scripts/download_libero_pro.sh
#   MAX_WORKERS=4 FORCE=1 bash vla-scripts/download_libero_pro.sh
#   HF_ENDPOINT=https://huggingface.co bash vla-scripts/download_libero_pro.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIBERO_PRO_PKG="${ROOT_DIR}/third_party/LIBERO-PRO/libero/libero"
BDDL_DIR="${LIBERO_PRO_PKG}/bddl_files"
INIT_DIR="${LIBERO_PRO_PKG}/init_files"
LOCAL_DIR="${LOCAL_DIR:-/tmp/libero_pro_data}"
MAX_WORKERS="${MAX_WORKERS:-4}"
FORCE="${FORCE:-0}"
MAX_RETRIES="${MAX_RETRIES:-12}"
RETRY_SLEEP_SEC="${RETRY_SLEEP_SEC:-30}"

# Prefer mirror; override with HF_ENDPOINT=https://huggingface.co if needed.
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-300}"

# HF dataset ships lan/object/swap/task (no *_env). Default eval uses swap.
MARKER_SUITE="${MARKER_SUITE:-libero_spatial_swap}"

download_ready() {
  [[ -d "${LOCAL_DIR}/bddl_files/${MARKER_SUITE}" && -d "${LOCAL_DIR}/init_files/${MARKER_SUITE}" ]] \
    && [[ -d "${LOCAL_DIR}/init_files" ]] \
    && [[ "$(find "${LOCAL_DIR}/init_files/${MARKER_SUITE}" -name '*.pruned_init' 2>/dev/null | wc -l)" -ge 1 ]]
}

installed_ready() {
  [[ -d "${BDDL_DIR}/${MARKER_SUITE}" && -d "${INIT_DIR}/${MARKER_SUITE}" ]] \
    && [[ "$(find "${INIT_DIR}/${MARKER_SUITE}" -name '*.pruned_init' 2>/dev/null | wc -l)" -ge 1 ]]
}

if installed_ready && [[ "${FORCE}" != "1" ]]; then
  echo "LIBERO-Pro suites already present (found ${MARKER_SUITE})."
  echo "  bddl: ${BDDL_DIR}/${MARKER_SUITE}"
  echo "  init: ${INIT_DIR}/${MARKER_SUITE}"
  du -sh "${BDDL_DIR}" "${INIT_DIR}"
  exit 0
fi

if [[ ! -d "${ROOT_DIR}/third_party/LIBERO-PRO" ]]; then
  echo "ERROR: missing ${ROOT_DIR}/third_party/LIBERO-PRO"
  echo "Clone first: git clone https://github.com/Zxy-MLlab/LIBERO-PRO.git third_party/LIBERO-PRO"
  exit 1
fi

mkdir -p "${LOCAL_DIR}" "${BDDL_DIR}" "${INIT_DIR}"

# hf download may soft-return on 429 without failing; retry until marker suite exists.
attempt=1
while ! download_ready; do
  if (( attempt > MAX_RETRIES )); then
    echo "ERROR: download incomplete after ${MAX_RETRIES} attempts."
    echo "Expected: ${LOCAL_DIR}/{bddl_files,init_files}/${MARKER_SUITE}"
    echo "Local snapshot so far:"
    ls "${LOCAL_DIR}" || true
    ls "${LOCAL_DIR}/bddl_files" 2>/dev/null | head -40 || true
    exit 1
  fi

  echo "Downloading zhouxueyang/LIBERO-Pro via ${HF_ENDPOINT}"
  echo "  attempt=${attempt}/${MAX_RETRIES} max-workers=${MAX_WORKERS} local-dir=${LOCAL_DIR}"
  set +e
  hf_out="$(hf download zhouxueyang/LIBERO-Pro \
    --repo-type dataset \
    --local-dir "${LOCAL_DIR}" \
    --max-workers "${MAX_WORKERS}" 2>&1)"
  hf_rc=$?
  set -e
  printf '%s\n' "${hf_out}"

  if download_ready; then
    break
  fi

  if printf '%s' "${hf_out}" | grep -q '429'; then
    echo "Rate-limited (429). Sleeping ${RETRY_SLEEP_SEC}s before retry..."
  elif (( hf_rc != 0 )); then
    echo "hf download failed (rc=${hf_rc}). Sleeping ${RETRY_SLEEP_SEC}s before retry..."
  else
    echo "Download returned but marker suite missing. Sleeping ${RETRY_SLEEP_SEC}s before retry..."
  fi
  sleep "${RETRY_SLEEP_SEC}"
  attempt=$((attempt + 1))
done

echo "Installing into ${LIBERO_PRO_PKG} ..."
# Same layout as upstream LIBERO-PRO README:
#   mv libero_data/bddl_files/* libero/libero/bddl_files/
#   mv libero_data/init_files/* libero/libero/init_files/
cp -a "${LOCAL_DIR}/bddl_files/." "${BDDL_DIR}/"
cp -a "${LOCAL_DIR}/init_files/." "${INIT_DIR}/"

if ! installed_ready; then
  echo "ERROR: install failed; expected suite '${MARKER_SUITE}' under bddl/init"
  echo "  bddl suites sample:"
  ls "${BDDL_DIR}" | head -30
  echo "  init suites sample:"
  ls "${INIT_DIR}" | head -30
  exit 1
fi

echo "Done. LIBERO-Pro suites ready at:"
echo "  ${BDDL_DIR}"
echo "  ${INIT_DIR}"
du -sh "${BDDL_DIR}" "${INIT_DIR}"
echo
echo "Next: bash vla-scripts/run_libero_pro_eval.sh"
