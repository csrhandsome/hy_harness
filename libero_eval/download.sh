#!/usr/bin/env bash
# Download LIBERO benchmark assets and legacy VLA-Adapter Pro checkpoints.
#
# Usage:
#   bash libero_eval/download.sh                    # Plus + Pro benchmark assets
#   bash libero_eval/download.sh plus pro           # selected benchmark assets
#   bash libero_eval/download.sh pro-ckpts           # legacy VLA-Adapter checkpoints
#   bash libero_eval/download.sh all                 # assets and legacy checkpoints
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAX_WORKERS="${MAX_WORKERS:-4}"
FORCE="${FORCE:-0}"

# Prefer mirror; override with HF_ENDPOINT=https://huggingface.co if needed.
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-300}"

usage() {
  cat <<'EOF'
Usage: bash libero_eval/download.sh [plus|pro|pro-ckpts|all]...

Without a target, this downloads LIBERO-plus assets and LIBERO-Pro BDDL/init
files. pro-ckpts downloads legacy VLA-Adapter checkpoints; they are not
Hy-VLA checkpoints and must not be used as CKPT_PATH.

Environment overrides: HF_ENDPOINT, MAX_WORKERS, FORCE, PLUS_LOCAL_DIR,
PRO_LOCAL_DIR, OUT_DIR, ONLY, MAX_RETRIES, RETRY_SLEEP_SEC.
EOF
}

require_dir() {
  local path="$1"
  local clone_command="$2"
  if [[ ! -d "${path}" ]]; then
    echo "ERROR: missing ${path}"
    echo "Clone first: ${clone_command}"
    exit 1
  fi
}

download_plus() {
  local pkg="${ROOT_DIR}/third_party/LIBERO-plus/libero/libero"
  local assets="${pkg}/assets"
  local local_dir="${PLUS_LOCAL_DIR:-/tmp/libero_plus_hf}"

  if [[ -d "${assets}/new_objects" && "${FORCE}" != "1" ]]; then
    echo "LIBERO-plus assets already present at: ${assets}"
    du -sh "${assets}"
    return
  fi
  require_dir "${ROOT_DIR}/third_party/LIBERO-plus" \
    "git clone https://github.com/sylvestf/LIBERO-plus.git third_party/LIBERO-plus"
  mkdir -p "${local_dir}"

  echo "Downloading LIBERO-plus assets via ${HF_ENDPOINT} (max-workers=${MAX_WORKERS}) ..."
  hf download Sylvest/LIBERO-plus assets.zip \
    --repo-type dataset --local-dir "${local_dir}" --max-workers "${MAX_WORKERS}"
  echo "Unzipping into ${pkg} ..."
  unzip -q -o "${local_dir}/assets.zip" -d "${pkg}"

  # The archive may nest assets under inspire/.../LIBERO-plus-0/assets.
  if [[ ! -d "${assets}/new_objects" ]]; then
    local nested
    nested="$(find "${pkg}" -type d -path '*/LIBERO-plus-0/assets' -print -quit 2>/dev/null || true)"
    if [[ -n "${nested}" ]]; then
      rm -rf "${assets}"
      mv "${nested}" "${assets}"
      rm -rf "${pkg}/inspire"
    fi
  fi
  if [[ ! -d "${assets}/new_objects" ]]; then
    echo "ERROR: assets install failed; expected ${assets}/new_objects"
    exit 1
  fi
  echo "Done. LIBERO-plus assets ready at: ${assets}"
  du -sh "${assets}"
}

pro_download_ready() {
  local local_dir="$1" marker="$2"
  [[ -d "${local_dir}/bddl_files/${marker}" && -d "${local_dir}/init_files/${marker}" ]] && \
    [[ -n "$(find "${local_dir}/init_files/${marker}" -name '*.pruned_init' -print -quit 2>/dev/null)" ]]
}

pro_installed_ready() {
  local bddl="$1" init="$2" marker="$3"
  [[ -d "${bddl}/${marker}" && -d "${init}/${marker}" ]] && \
    [[ -n "$(find "${init}/${marker}" -name '*.pruned_init' -print -quit 2>/dev/null)" ]]
}

download_pro() {
  local pkg="${ROOT_DIR}/third_party/LIBERO-PRO/libero/libero"
  local bddl="${pkg}/bddl_files" init="${pkg}/init_files"
  local local_dir="${PRO_LOCAL_DIR:-/tmp/libero_pro_data}"
  local marker="${MARKER_SUITE:-libero_spatial_swap}"
  local max_retries="${MAX_RETRIES:-12}" retry_sleep_sec="${RETRY_SLEEP_SEC:-30}"
  local attempt=1 hf_out hf_rc

  if pro_installed_ready "${bddl}" "${init}" "${marker}" && [[ "${FORCE}" != "1" ]]; then
    echo "LIBERO-Pro suites already present (found ${marker})."
    du -sh "${bddl}" "${init}"
    return
  fi
  require_dir "${ROOT_DIR}/third_party/LIBERO-PRO" \
    "git clone https://github.com/Zxy-MLlab/LIBERO-PRO.git third_party/LIBERO-PRO"
  mkdir -p "${local_dir}" "${bddl}" "${init}"

  # hf download can soft-return on 429; retry until an expected suite exists.
  while ! pro_download_ready "${local_dir}" "${marker}"; do
    if (( attempt > max_retries )); then
      echo "ERROR: LIBERO-Pro download incomplete after ${max_retries} attempts."
      echo "Expected: ${local_dir}/{bddl_files,init_files}/${marker}"
      exit 1
    fi
    echo "Downloading zhouxueyang/LIBERO-Pro via ${HF_ENDPOINT}"
    echo "  attempt=${attempt}/${max_retries} max-workers=${MAX_WORKERS} local-dir=${local_dir}"
    set +e
    hf_out="$(hf download zhouxueyang/LIBERO-Pro --repo-type dataset \
      --local-dir "${local_dir}" --max-workers "${MAX_WORKERS}" 2>&1)"
    hf_rc=$?
    set -e
    printf '%s\n' "${hf_out}"
    if pro_download_ready "${local_dir}" "${marker}"; then
      break
    fi
    if printf '%s' "${hf_out}" | grep -q '429'; then
      echo "Rate-limited (429). Sleeping ${retry_sleep_sec}s before retry..."
    elif (( hf_rc != 0 )); then
      echo "hf download failed (rc=${hf_rc}). Sleeping ${retry_sleep_sec}s before retry..."
    else
      echo "Download returned but marker suite missing. Sleeping ${retry_sleep_sec}s before retry..."
    fi
    sleep "${retry_sleep_sec}"
    attempt=$((attempt + 1))
  done

  echo "Installing LIBERO-Pro suites into ${pkg} ..."
  cp -a "${local_dir}/bddl_files/." "${bddl}/"
  cp -a "${local_dir}/init_files/." "${init}/"
  if ! pro_installed_ready "${bddl}" "${init}" "${marker}"; then
    echo "ERROR: install failed; expected suite '${marker}' under bddl/init"
    exit 1
  fi
  echo "Done. LIBERO-Pro suites ready at: ${bddl} and ${init}"
  du -sh "${bddl}" "${init}"
}

ckpt_ready() {
  local dir="$1"
  [[ -f "${dir}/config.json" ]] && \
    [[ -n "$(find "${dir}" -maxdepth 1 -type f \
      \( -name '*.safetensors' -o -name '*.bin' -o -name '*.pt' \) -print -quit 2>/dev/null)" ]]
}

download_pro_ckpts() {
  local out_dir="${OUT_DIR:-${ROOT_DIR}/outputs}" only="${ONLY:-}"
  local entry suite repo local_name dest
  local -a ckpts=(
    "libero_spatial|VLA-Adapter/LIBERO-Spatial-Pro|LIBERO-Spatial-Pro"
    "libero_object|VLA-Adapter/LIBERO-Object-Pro|LIBERO-Object-Pro"
    "libero_goal|VLA-Adapter/LIBERO-Goal-Pro|LIBERO-Goal-Pro"
    "libero_10|VLA-Adapter/LIBERO-Long-Pro|LIBERO-Long-Pro"
  )
  mkdir -p "${out_dir}"
  for entry in "${ckpts[@]}"; do
    IFS='|' read -r suite repo local_name <<<"${entry}"
    if [[ -n "${only}" && "${only}" != "${suite}" && "${only}" != "${local_name}" ]]; then
      continue
    fi
    dest="${out_dir}/${local_name}"
    if ckpt_ready "${dest}" && [[ "${FORCE}" != "1" ]]; then
      echo "Skip (already present): ${dest}"
      du -sh "${dest}"
      continue
    fi
    echo "Downloading ${repo} → ${dest} via ${HF_ENDPOINT} ..."
    hf download "${repo}" --local-dir "${dest}" --max-workers "${MAX_WORKERS}"
    if ! ckpt_ready "${dest}"; then
      echo "ERROR: download incomplete for ${dest}"
      exit 1
    fi
    echo "Done: ${dest}"
    du -sh "${dest}"
  done
}

declare -A targets=()
if (( $# == 0 )); then
  targets[plus]=1
  targets[pro]=1
else
  for target in "$@"; do
    case "${target}" in
      plus|pro|pro-ckpts) targets["${target}"]=1 ;;
      all) targets[plus]=1; targets[pro]=1; targets[pro-ckpts]=1 ;;
      -h|--help) usage; exit 0 ;;
      *) echo "ERROR: unknown target '${target}'" >&2; usage >&2; exit 2 ;;
    esac
  done
fi
[[ -n "${targets[plus]:-}" ]] && download_plus
[[ -n "${targets[pro]:-}" ]] && download_pro
[[ -n "${targets[pro-ckpts]:-}" ]] && download_pro_ckpts
