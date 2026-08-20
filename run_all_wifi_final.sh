#!/bin/bash
set -euo pipefail

cd /Users/kushagraaitha/Documents/ReVo

OUT="output/wifi_30video_eval_final"
LOGDIR="$OUT/batch_logs"
mkdir -p "$LOGDIR"

for METHOD in baseline abr gcc; do
    echo
    echo "=================================================="
    echo "STARTING $METHOD — $(date)"
    echo "=================================================="

    .venv/bin/python scripts/wifi_eval_runner.py \
        --method "$METHOD" \
        --full_run \
        --output_root "$OUT" \
        2>&1 | tee "$LOGDIR/${METHOD}.log"

    echo
    echo "FINISHED $METHOD — $(date)"
done

echo
echo "=================================================="
echo "ALL 90 WIFI RUNS FINISHED — $(date)"
echo "=================================================="
