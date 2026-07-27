#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_VARIANT="final_estimator" \
  exec bash "$SCRIPT_DIR/run_wifi_trace14_two_container_transport_fix.sh"
