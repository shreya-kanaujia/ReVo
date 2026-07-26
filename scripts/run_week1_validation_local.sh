#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/output/week1_estimator/validation}"
PYTHON="${PYTHON:-$ROOT_DIR/.venv/bin/python}"
CODEC="${CODEC:-h265}"
RUN_SECONDS="${RUN_SECONDS:-60}"
APPLY_TC="${APPLY_TC:-1}"
VALIDATION_TRAIN_ONLY="${VALIDATION_TRAIN_ONLY:-0}"
VALIDATION_TRAIN_PACKET_SIZE="${VALIDATION_TRAIN_PACKET_SIZE:-1200}"
VALIDATION_TRAIN_PACKETS="${VALIDATION_TRAIN_PACKETS:-320}"
VALIDATION_TRAIN_INTERVAL="${VALIDATION_TRAIN_INTERVAL:-0.25}"
VALIDATION_PACED_PROBE_ONLY="${VALIDATION_PACED_PROBE_ONLY:-0}"
VALIDATION_PACED_BITRATE_MBPS="${VALIDATION_PACED_BITRATE_MBPS:-9}"
VALIDATION_PACED_PAYLOAD_SIZE="${VALIDATION_PACED_PAYLOAD_SIZE:-1024}"
VALIDATION_PACED_SOFT_BUFFER_LIMIT="${VALIDATION_PACED_SOFT_BUFFER_LIMIT:-32768}"
VALIDATION_PACED_BACKPRESSURE_SLEEP="${VALIDATION_PACED_BACKPRESSURE_SLEEP:-0.002}"
ICE_CONSENT_TIMEOUT_S="${ICE_CONSENT_TIMEOUT_S:-5}"
ENABLE_DIAGNOSTICS="${ENABLE_DIAGNOSTICS:-1}"
TRACE_PATH="${TRACE_PATH:-$ROOT_DIR/src/sender/traces/week1_step_8_3_6_60s.log}"
TRACE_NAME="${TRACE_NAME:-step}"
GT_CSV="${GT_CSV:-$OUT_DIR/ground_truth_${TRACE_NAME}.csv}"
CAPACITY_CSV="${CAPACITY_CSV:-$OUT_DIR/capacity_estimator_${TRACE_NAME}.csv}"
MEASUREMENT_CSV="${MEASUREMENT_CSV:-$OUT_DIR/sender_frame_measurements_${TRACE_NAME}.csv}"
PROBE_SENDER_CSV="${PROBE_SENDER_CSV:-$OUT_DIR/probe_sender_${TRACE_NAME}.csv}"
SENDER_DIAGNOSTIC_CSV="${SENDER_DIAGNOSTIC_CSV:-$OUT_DIR/sender_diagnostics_${TRACE_NAME}.csv}"
RECEIVER_DIAGNOSTIC_CSV="${RECEIVER_DIAGNOSTIC_CSV:-$OUT_DIR/receiver_diagnostics_${TRACE_NAME}.csv}"
OVERLAY_PNG="${OVERLAY_PNG:-$OUT_DIR/capacity_overlay_${TRACE_NAME}.png}"
MAX_FRAME_PNG="${MAX_FRAME_PNG:-$OUT_DIR/max_frame_size_${TRACE_NAME}.png}"
METRICS_REPORT="${METRICS_REPORT:-$OUT_DIR/capacity_metrics_${TRACE_NAME}.txt}"
RGB_INPUT="${RGB_INPUT:-$OUT_DIR/input/rgb_${RUN_SECONDS}s.mp4}"
DEPTH_INPUT="${DEPTH_INPUT:-$OUT_DIR/input/depth_${RUN_SECONDS}s.mp4}"
RGB_SOURCE="${RGB_SOURCE:-$ROOT_DIR/data/gt_rgb/1lSejjfNHpw_0075_S0_E728_L671_T47_R1471_B847.mp4}"
DEPTH_SOURCE="${DEPTH_SOURCE:-$ROOT_DIR/data/gt_depth/1lSejjfNHpw_0075_S0_E728_L671_T47_R1471_B847_vis.mp4}"
IFACE="${IFACE:-}"

mkdir -p "$OUT_DIR/input" "$OUT_DIR/logs"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "This validation requires Linux tc. Run inside a privileged Linux VM/container." >&2
  exit 1
fi

if [[ ! -x "$PYTHON" ]]; then
  echo "Python not found/executable at $PYTHON" >&2
  exit 1
fi

if [[ ! -f "$RGB_INPUT" ]]; then
  ffmpeg -y -stream_loop -1 -i "$RGB_SOURCE" -t "$RUN_SECONDS" -c copy "$RGB_INPUT"
fi
if [[ ! -f "$DEPTH_INPUT" ]]; then
  ffmpeg -y -stream_loop -1 -i "$DEPTH_SOURCE" -t "$RUN_SECONDS" -c copy "$DEPTH_INPUT"
fi

if [[ -z "$IFACE" ]]; then
  IFACE="$(ip -o route show default | awk '{for (i=1;i<=NF;i++) if ($i=="dev") {print $(i+1); exit}}')"
fi

HOST_IP="$(ip -4 addr show "$IFACE" | awk '/inet / {print $2}' | cut -d/ -f1 | head -1)"
if [[ -z "$HOST_IP" ]]; then
  echo "Could not determine IPv4 address for $IFACE" >&2
  exit 1
fi

cleanup() {
  set +e
  if [[ -n "${TRACE_PID:-}" ]]; then kill "$TRACE_PID" 2>/dev/null; wait "$TRACE_PID" 2>/dev/null; fi
  if [[ -n "${SENDER_PID:-}" ]]; then kill "$SENDER_PID" 2>/dev/null; wait "$SENDER_PID" 2>/dev/null; fi
  if [[ -n "${RECEIVER_PID:-}" ]]; then kill "$RECEIVER_PID" 2>/dev/null; wait "$RECEIVER_PID" 2>/dev/null; fi
  if [[ -n "${SIGNAL_PID:-}" ]]; then kill "$SIGNAL_PID" 2>/dev/null; wait "$SIGNAL_PID" 2>/dev/null; fi
  tc qdisc del dev "$IFACE" root >/dev/null 2>&1 || true
}
trap cleanup EXIT

"$PYTHON" "$ROOT_DIR/src/signalling_server.py" > "$OUT_DIR/logs/signaling_${TRACE_NAME}.log" 2>&1 &
SIGNAL_PID=$!
sleep 1

(
  cd "$ROOT_DIR/src/receiver"
  RECEIVER_EXTRA_ARGS=(--ice_consent_timeout_s "$ICE_CONSENT_TIMEOUT_S")
  if [[ "$ENABLE_DIAGNOSTICS" == "1" ]]; then
    RECEIVER_EXTRA_ARGS+=(
      --diagnostic_csv "$RECEIVER_DIAGNOSTIC_CSV"
      --diagnostic_ice
    )
  fi
  "$PYTHON" receiver-3d.py \
    --out "$OUT_DIR/receiver_${TRACE_NAME}_rgb.mp4" \
    --out_depth "$OUT_DIR/receiver_${TRACE_NAME}_depth.mp4" \
    --server_ip "$HOST_IP" \
    --codec "$CODEC" \
    --estimator_alpha 0.1 \
    "${RECEIVER_EXTRA_ARGS[@]}"
) > "$OUT_DIR/logs/receiver_${TRACE_NAME}.log" 2>&1 &
RECEIVER_PID=$!
sleep 2

if [[ "$APPLY_TC" == "1" ]]; then
  "$PYTHON" "$ROOT_DIR/src/sender/run_loss_trace.py" \
    --trace "$TRACE_PATH" \
    --interface "$IFACE" \
    --ground_truth_csv "$GT_CSV" \
    --duration "$RUN_SECONDS" \
    --loss_override 0.0 \
    --delay_override_ms 0 \
    > "$OUT_DIR/logs/tc_${TRACE_NAME}.log" 2>&1 &
  TRACE_PID=$!
  sleep 1
else
  printf 'monotonic_timestamp,elapsed_s,trace_time_s,bandwidth_mbps,loss_ratio,delay_ms,interface\n' > "$GT_CSV"
  printf '%.9f,0.000000000,0.000000000,0.000000000,0.000000000,0,%s\n' "$(python3 - <<'PY'
import time
print(f"{time.perf_counter():.9f}")
PY
)" "$IFACE" >> "$GT_CSV"
fi

(
  cd "$ROOT_DIR/src/sender"
  EXTRA_ARGS=(--ice_consent_timeout_s "$ICE_CONSENT_TIMEOUT_S")
  if [[ "$ENABLE_DIAGNOSTICS" == "1" ]]; then
    EXTRA_ARGS+=(
      --diagnostic_csv "$SENDER_DIAGNOSTIC_CSV"
      --diagnostic_ice
    )
  fi
  if [[ "$VALIDATION_TRAIN_ONLY" == "1" ]]; then
    EXTRA_ARGS+=(
      --validation_train_only
      --validation_train_duration "$RUN_SECONDS"
      --validation_train_packet_size "$VALIDATION_TRAIN_PACKET_SIZE"
      --validation_train_packets "$VALIDATION_TRAIN_PACKETS"
      --validation_train_interval "$VALIDATION_TRAIN_INTERVAL"
    )
  fi
  if [[ "$VALIDATION_PACED_PROBE_ONLY" == "1" ]]; then
    EXTRA_ARGS+=(
      --validation_paced_probe_only
      --validation_paced_duration "$RUN_SECONDS"
      --validation_paced_bitrate_mbps "$VALIDATION_PACED_BITRATE_MBPS"
      --validation_paced_payload_size "$VALIDATION_PACED_PAYLOAD_SIZE"
      --validation_paced_soft_buffer_limit "$VALIDATION_PACED_SOFT_BUFFER_LIMIT"
      --validation_paced_backpressure_sleep "$VALIDATION_PACED_BACKPRESSURE_SLEEP"
      --probe_sender_csv "$PROBE_SENDER_CSV"
    )
  fi
  "$PYTHON" sender-3d.py \
    --file "$RGB_INPUT" \
    --depth_file "$DEPTH_INPUT" \
    --server_ip "$HOST_IP" \
    --codec "$CODEC" \
    --measurement_csv "$MEASUREMENT_CSV" \
    --capacity_csv "$CAPACITY_CSV" \
    "${EXTRA_ARGS[@]}"
) > "$OUT_DIR/logs/sender_${TRACE_NAME}.log" 2>&1 &
SENDER_PID=$!

set +e
wait "$SENDER_PID"
SENDER_STATUS=$?
set -e
SENDER_PID=""
wait "$RECEIVER_PID" || true
RECEIVER_PID=""
if [[ -n "${TRACE_PID:-}" ]]; then
  kill "$TRACE_PID" 2>/dev/null || true
  wait "$TRACE_PID" 2>/dev/null || true
  TRACE_PID=""
fi

test -s "$GT_CSV"
test -s "$CAPACITY_CSV"
ANALYSIS_ARGS=(
  --ground-truth "$GT_CSV"
  --estimator "$CAPACITY_CSV"
  --plot "$OVERLAY_PNG"
  --max-frame-plot "$MAX_FRAME_PNG"
  --report "$METRICS_REPORT"
  --expect-duration-s "$RUN_SECONDS"
  --sender-log "$OUT_DIR/logs/sender_${TRACE_NAME}.log"
  --receiver-log "$OUT_DIR/logs/receiver_${TRACE_NAME}.log"
)
if [[ "$VALIDATION_PACED_PROBE_ONLY" == "1" ]]; then
  test -s "$PROBE_SENDER_CSV"
  ANALYSIS_ARGS+=(--probe-sender "$PROBE_SENDER_CSV")
fi

set +e
MPLCONFIGDIR="$OUT_DIR/.mplconfig" "$PYTHON" \
  "$ROOT_DIR/scripts/analyze_capacity_estimator.py" "${ANALYSIS_ARGS[@]}"
ANALYSIS_STATUS=$?
set -e

echo "validation data collection finished:"
echo "  ground truth: $GT_CSV"
echo "  estimator:    $CAPACITY_CSV"
if [[ "$VALIDATION_PACED_PROBE_ONLY" == "1" ]]; then
  echo "  probe sender: $PROBE_SENDER_CSV"
fi
echo "  report:       $METRICS_REPORT"
echo "  overlay:      $OVERLAY_PNG"

if [[ "$SENDER_STATUS" -ne 0 ]]; then
  echo "sender exited with status $SENDER_STATUS; validation failed" >&2
  exit "$SENDER_STATUS"
fi
if [[ "$ANALYSIS_STATUS" -ne 0 ]]; then
  echo "validation completeness/analysis failed with status $ANALYSIS_STATUS" >&2
  exit "$ANALYSIS_STATUS"
fi
echo "validation run passed completeness checks"
