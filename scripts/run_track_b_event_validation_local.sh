#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${OUT_DIR:?Set OUT_DIR to a new non-existing result directory}"

MODE=combined \
CASE=no_impairment \
RUN_SECONDS="${EVENT_TIMEOUT_S:-60}" \
EVENT_BASED_LOCAL=1 \
EVENT_TIMEOUT_S="${EVENT_TIMEOUT_S:-60}" \
OUT_DIR="$OUT_DIR" \
bash "$ROOT_DIR/scripts/run_track_b_two_container.sh"
