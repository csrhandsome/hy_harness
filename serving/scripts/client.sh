#!/usr/bin/env bash
set -euo pipefail
SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPTS/_run.sh" client "$@"
