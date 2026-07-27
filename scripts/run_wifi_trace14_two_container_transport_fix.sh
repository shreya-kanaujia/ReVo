#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_VARIANT="${RUN_VARIANT:-two_container_transport_fix}"
OUT_DIR="$ROOT_DIR/output/week1_estimator/loss_robustness/wifi_trace14_${RUN_VARIANT}"
CONTAINER_OUT="/revo/output/week1_estimator/loss_robustness/wifi_trace14_${RUN_VARIANT}"
CONTAINER_TRACE="/revo/src/sender/traces/wifi/trace14.log"
CONTAINER_INPUT="/revo/output/week1_estimator/paced_probe_validation/input"
NETWORK="revo_wifi_trace14_${RUN_VARIANT}"
SENDER_CONTAINER="revo-trace14-${RUN_VARIANT//_/-}-sender"
RECEIVER_CONTAINER="revo-trace14-${RUN_VARIANT//_/-}-receiver"
IMAGE="revo-week1-validation:local"
START_SIGNAL="$OUT_DIR/START"
RUN_SECONDS="324.15"

if [[ -e "$OUT_DIR" ]]; then
  echo "Refusing to overwrite existing validation output: $OUT_DIR" >&2
  exit 2
fi

mkdir -p "$OUT_DIR/logs" "$OUT_DIR/metadata"

SIGNALING_PID=""
SENDER_PID=""
RECEIVER_PID=""
TRACE_PID=""

cleanup() {
  set +e
  [[ -n "$TRACE_PID" ]] && kill "$TRACE_PID" 2>/dev/null
  [[ -n "$SENDER_PID" ]] && kill "$SENDER_PID" 2>/dev/null
  [[ -n "$RECEIVER_PID" ]] && kill "$RECEIVER_PID" 2>/dev/null
  [[ -n "$SIGNALING_PID" ]] && kill "$SIGNALING_PID" 2>/dev/null
  docker rm -f "$SENDER_CONTAINER" "$RECEIVER_CONTAINER" >/dev/null 2>&1
  docker network rm "$NETWORK" >/dev/null 2>&1
}
trap cleanup EXIT INT TERM

wait_for_log() {
  local path="$1"
  local pattern="$2"
  local attempts=0
  while ! grep -q "$pattern" "$path" 2>/dev/null; do
    attempts=$((attempts + 1))
    if (( attempts > 900 )); then
      echo "Timed out waiting for '$pattern' in $path" >&2
      return 1
    fi
    sleep 0.1
  done
}

docker network create --driver bridge --subnet 172.30.14.0/24 "$NETWORK" \
  >"$OUT_DIR/metadata/docker_network_create.txt"
docker run -d --name "$SENDER_CONTAINER" --privileged \
  --network "$NETWORK" --ip 172.30.14.10 \
  --add-host host.docker.internal:host-gateway \
  -v "$ROOT_DIR:/revo" "$IMAGE" sleep infinity \
  >"$OUT_DIR/metadata/sender_container_id.txt"
docker run -d --name "$RECEIVER_CONTAINER" \
  --network "$NETWORK" --ip 172.30.14.20 \
  --add-host host.docker.internal:host-gateway \
  -v "$ROOT_DIR:/revo" "$IMAGE" sleep infinity \
  >"$OUT_DIR/metadata/receiver_container_id.txt"

docker network inspect "$NETWORK" \
  >"$OUT_DIR/metadata/docker_network_inspect.json"
docker inspect "$SENDER_CONTAINER" "$RECEIVER_CONTAINER" \
  >"$OUT_DIR/metadata/docker_inspect.json"
docker exec "$SENDER_CONTAINER" ip -brief address \
  >"$OUT_DIR/metadata/sender_interfaces.txt"
docker exec "$RECEIVER_CONTAINER" ip -brief address \
  >"$OUT_DIR/metadata/receiver_interfaces.txt"
docker exec "$SENDER_CONTAINER" ip route get 172.30.14.20 \
  >"$OUT_DIR/metadata/sender_route_to_receiver.txt"

docker exec "$RECEIVER_CONTAINER" sh -lc \
  "cd /revo && exec /usr/local/bin/python3 src/signalling_server.py" \
  >"$OUT_DIR/logs/signaling.log" 2>&1 &
SIGNALING_PID=$!
sleep 1

docker exec "$RECEIVER_CONTAINER" sh -lc \
  "cd /revo && exec /usr/local/bin/python3 src/receiver/receiver-3d.py \
    --out '$CONTAINER_OUT/receiver_rgb.mp4' \
    --out_depth '$CONTAINER_OUT/receiver_depth.mp4' \
    --server_ip 172.30.14.20 \
    --stun stun:stun.l.google.com:19302 \
    --codec h265 \
    --receiver_capacity_csv '$CONTAINER_OUT/receiver_arrivals_estimator.csv' \
    --diagnostic_csv '$CONTAINER_OUT/receiver_diagnostics.csv' \
    --diagnostic_ice \
    --ice_consent_timeout_s 5 \
    --validation_sctp_gap_rtt_fix" \
  >"$OUT_DIR/logs/receiver.log" 2>&1 &
RECEIVER_PID=$!
sleep 1

docker exec "$SENDER_CONTAINER" sh -lc \
  "cd /revo && exec /usr/local/bin/python3 src/sender/sender-3d.py \
    --file '$CONTAINER_INPUT/rgb_60s.mp4' \
    --depth_file '$CONTAINER_INPUT/depth_60s.mp4' \
    --server_ip 172.30.14.20 \
    --stun_url stun:stun.l.google.com:19302 \
    --codec h265 \
    --measurement_csv '$CONTAINER_OUT/sender_frame_measurements.csv' \
    --capacity_csv '$CONTAINER_OUT/sender_feedback.csv' \
    --probe_sender_csv '$CONTAINER_OUT/sender_probe.csv' \
    --diagnostic_csv '$CONTAINER_OUT/sender_diagnostics.csv' \
    --diagnostic_ice \
    --ice_consent_timeout_s 5 \
    --validation_sctp_gap_rtt_fix \
    --validation_paced_probe_only \
    --validation_paced_duration '$RUN_SECONDS' \
    --validation_paced_bitrate_mbps 9 \
    --validation_paced_payload_size 1024 \
    --validation_paced_soft_buffer_limit 32768 \
    --validation_paced_backpressure_sleep 0.002 \
    --validation_start_signal_path '$CONTAINER_OUT/START'" \
  >"$OUT_DIR/logs/sender.log" 2>&1 &
SENDER_PID=$!

docker exec "$SENDER_CONTAINER" sh -lc \
  "cd /revo && exec /usr/local/bin/python3 -u src/sender/run_loss_trace.py \
    --trace '$CONTAINER_TRACE' \
    --interface eth0 \
    --ground_truth_csv '$CONTAINER_OUT/applied_trace.csv' \
    --duration '$RUN_SECONDS' \
    --delay_override_ms 0 \
    --start_signal_path '$CONTAINER_OUT/START'" \
  >"$OUT_DIR/logs/tc.log" 2>&1 &
TRACE_PID=$!

wait_for_log "$OUT_DIR/logs/sender.log" "waiting for validation start signal"
wait_for_log "$OUT_DIR/logs/tc.log" "Waiting for validation start signal"
touch "$START_SIGNAL"

sleep 3
docker exec "$SENDER_CONTAINER" tc -s qdisc show dev eth0 \
  >"$OUT_DIR/metadata/tc_after_3s.txt"

if ! grep -Eq "CandidatePair\\(\\('172\\.30\\.14\\.10'.*'172\\.30\\.14\\.20'.*SUCCEEDED|CandidatePair\\(\\('172\\.30\\.14\\.20'.*'172\\.30\\.14\\.10'.*SUCCEEDED" \
    "$OUT_DIR/logs/sender.log" "$OUT_DIR/logs/receiver.log"; then
  echo "Preflight failed: cross-container ICE pair was not selected" >&2
  exit 3
fi
if ! awk -F, 'NR > 1 && ($5 + 0) > 0 { found=1; exit } END { exit !found }' \
    "$OUT_DIR/applied_trace.csv"; then
  echo "Preflight failed: applied trace has no nonzero loss" >&2
  exit 3
fi
if ! awk -F, 'NR > 2 && ($2 + 0) > previous + 1 { found=1; exit } NR > 1 { previous=$2 + 0 } END { exit !found }' \
    "$OUT_DIR/receiver_arrivals_estimator.csv"; then
  echo "Preflight failed: receiver has no real sequence gap" >&2
  exit 3
fi
if ! awk '/dropped [1-9][0-9]*/ { found=1 } END { exit !found }' \
    "$OUT_DIR/metadata/tc_after_3s.txt"; then
  echo "Preflight failed: tc drop counter did not increase" >&2
  exit 3
fi

{
  echo "Two-container wifi/trace14 $RUN_VARIANT preflight verification"
  echo
  echo "Docker network: $NETWORK (172.30.14.0/24)"
  echo "Sender container: $SENDER_CONTAINER, 172.30.14.10"
  echo "Receiver container: $RECEIVER_CONTAINER, 172.30.14.20"
  echo "Sender route:"
  sed 's/^/  /' "$OUT_DIR/metadata/sender_route_to_receiver.txt"
  echo "First nonzero applied loss rows:"
  awk -F, 'NR > 1 && ($5 + 0) > 0 { print "  " $0; count++ } count == 5 { exit }' \
    "$OUT_DIR/applied_trace.csv"
  echo "First receiver sequence gaps:"
  awk -F, 'NR > 2 && ($2 + 0) > previous + 1 { print "  " previous " -> " ($2 + 0); count++ } NR > 1 { previous=$2 + 0 } count == 5 { exit }' \
    "$OUT_DIR/receiver_arrivals_estimator.csv"
} >"$OUT_DIR/metadata/preflight_verification.txt"

set +e
wait "$TRACE_PID"
TRACE_STATUS=$?
TRACE_PID=""
wait "$SENDER_PID"
SENDER_STATUS=$?
SENDER_PID=""
wait "$RECEIVER_PID"
RECEIVER_STATUS=$?
RECEIVER_PID=""
set -e

printf '%s\n' "$TRACE_STATUS" >"$OUT_DIR/metadata/tc_exit_status.txt"
printf '%s\n' "$SENDER_STATUS" >"$OUT_DIR/metadata/sender_exit_status.txt"
printf '%s\n' "$RECEIVER_STATUS" >"$OUT_DIR/metadata/receiver_exit_status.txt"

if [[ "$TRACE_STATUS" -ne 0 || "$SENDER_STATUS" -ne 0 ]]; then
  echo "Validation failed: tc=$TRACE_STATUS sender=$SENDER_STATUS receiver=$RECEIVER_STATUS" >&2
  exit 4
fi

echo "Completed one $RUN_VARIANT two-container validation: $OUT_DIR"
