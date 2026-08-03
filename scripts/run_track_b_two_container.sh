#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${MODE:?Set MODE}"
CASE="${CASE:?Set CASE}"
TRACE_PATH="${TRACE_PATH:-}"
RUN_SECONDS="${RUN_SECONDS:?Set RUN_SECONDS}"
IMAGE="${IMAGE:-revo-week1-validation:local}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/output/track_b/$CASE/$MODE}"
RGB_SOURCE="${RGB_SOURCE:-$ROOT_DIR/data/gt_rgb/1lSejjfNHpw_0075_S0_E728_L671_T47_R1471_B847.mp4}"
DEPTH_SOURCE="${DEPTH_SOURCE:-$ROOT_DIR/data/gt_depth/1lSejjfNHpw_0075_S0_E728_L671_T47_R1471_B847_vis.mp4}"
TRACK_B_EXTRA_ARGS="${TRACK_B_EXTRA_ARGS:-}"
DIAGNOSTIC_CONTINUE_ON_MISS="${DIAGNOSTIC_CONTINUE_ON_MISS:-0}"
EVENT_BASED_LOCAL="${EVENT_BASED_LOCAL:-0}"
EVENT_TIMEOUT_S="${EVENT_TIMEOUT_S:-60}"
TRACK_B_CRITICAL_NICE="${TRACK_B_CRITICAL_NICE:--10}"
FROZEN_TRACK_B_ARGS="--track_b_high_qp 20 --track_b_low_qp 30 \
--track_b_buffer_soft_bytes 65536 --track_b_buffer_hard_bytes 131072 \
--track_b_buffer_growth_bytes 8192 --track_b_impairment_threshold 0.05 \
--track_b_severe_impairment_threshold 0.20 \
--track_b_capacity_safety_margin 0.85 \
--track_b_upgrade_headroom_fraction 0.15 \
--track_b_downgrade_confirmations 1 --track_b_upgrade_confirmations 3 \
--track_b_min_dwell_gops 2 --track_b_demand_alpha 0.2 \
--track_b_encode_lead_alpha 0.2 \
--receiver_health_freshness_s 2.5 --track_b_fps 0 \
--track_b_encoder_pool_threads 1 --track_b_encoder_process \
--track_b_encoder_max_inflight 2 --track_b_encoder_completed_lead 5 \
--track_b_sender_reserved_cpus 2 \
--track_b_shared_monotonic_clock \
--track_b_passive_service_floor_fraction 0.80 \
--track_b_probe_payload_bytes 1024 \
--track_b_probe_min_duration_s 0.40 --track_b_probe_max_duration_s 0.60 \
--track_b_probe_max_total_bytes 524288 \
--track_b_probe_max_additional_rate_mbps 20 \
--track_b_probe_max_rate_multiplier 8 \
--track_b_probe_min_confirmed_fraction 0.90 \
--track_b_probe_ack_timeout_s 0.25 --track_b_probe_cooldown_s 2.0 \
--track_b_probe_authorization_ttl_s 1.5 \
--track_b_probe_buffer_abort_bytes 131072 \
--track_b_probe_buffer_growth_abort_bytes 8192 \
--track_b_probe_impairment_abort_threshold 0.05 \
--track_b_probe_boundary_guard_s 0.08"

case "$MODE" in
  fixed_high|fixed_low|buffer_health|estimator_only|combined) ;;
  *) echo "Two-container Track B mode must be one of the five comparison modes" >&2; exit 2 ;;
esac
if [[ -e "$OUT_DIR" ]]; then
  echo "Refusing to overwrite $OUT_DIR" >&2
  exit 2
fi
if [[ "$CASE" != "no_impairment" && -z "$TRACE_PATH" ]]; then
  echo "Set TRACE_PATH for impaired cases" >&2
  exit 2
fi
if [[ "$EVENT_BASED_LOCAL" == "1" && ( "$MODE" != "combined" || "$CASE" != "no_impairment" ) ]]; then
  echo "Event-based local gate requires combined no_impairment mode" >&2
  exit 2
fi

SAFE_NAME="$(printf '%s-%s' "$CASE" "$MODE" | tr -cs 'a-zA-Z0-9' '-')"
NETWORK="revo-track-b-$SAFE_NAME"
SENDER_CONTAINER="$NETWORK-sender"
RECEIVER_CONTAINER="$NETWORK-receiver"
SIGNAL_CONTAINER="$NETWORK-signaling"
TRACE_CONTAINER="$NETWORK-trace"
CONTAINER_OUT="/revo/${OUT_DIR#"$ROOT_DIR/"}"
START_SIGNAL="$OUT_DIR/START"
STOP_SIGNAL="$OUT_DIR/STOP"
EVENT_STOP_ARG=""
DIAGNOSTIC_SENDER_ARGS=""
DIAGNOSTIC_VALIDATOR_ARG=""
if [[ "$EVENT_BASED_LOCAL" == "1" ]]; then
  EVENT_STOP_ARG="--validation_stop_signal_path '$CONTAINER_OUT/STOP'"
fi
if [[ "$DIAGNOSTIC_CONTINUE_ON_MISS" == "1" ]]; then
  DIAGNOSTIC_SENDER_ARGS="--diagnostic_continue_on_candidate_miss --candidate_diagnostic_csv '$CONTAINER_OUT/candidate_diagnostics.csv'"
  DIAGNOSTIC_VALIDATOR_ARG="--diagnostic-only"
fi
mkdir -p "$OUT_DIR/input" "$OUT_DIR/logs" "$OUT_DIR/metadata"

ffmpeg -loglevel error -y -stream_loop -1 -i "$RGB_SOURCE" \
  -t "$RUN_SECONDS" -c copy "$OUT_DIR/input/rgb.mp4"
ffmpeg -loglevel error -y -stream_loop -1 -i "$DEPTH_SOURCE" \
  -t "$RUN_SECONDS" -c copy "$OUT_DIR/input/depth.mp4"

sha256sum "$RGB_SOURCE" "$DEPTH_SOURCE" \
  >"$OUT_DIR/metadata/input_hashes.sha256"
if [[ -n "$TRACE_PATH" ]]; then
  sha256sum "$TRACE_PATH" >>"$OUT_DIR/metadata/input_hashes.sha256"
fi
{
  printf 'mode=%s\ncase=%s\nrun_seconds=%s\n' "$MODE" "$CASE" "$RUN_SECONDS"
  printf 'trace=%s\nimage=%s\nfrozen_track_b_args=%s\ntrack_b_extra_args=%s\n' \
    "$TRACE_PATH" "$IMAGE" "$FROZEN_TRACK_B_ARGS" "$TRACK_B_EXTRA_ARGS"
  printf 'receiver_decode_guard_s=0.010\n'
  printf 'event_based_local=%s\nevent_timeout_s=%s\n' \
    "$EVENT_BASED_LOCAL" "$EVENT_TIMEOUT_S"
  printf 'diagnostic_continue_on_candidate_miss=%s\n' \
    "$DIAGNOSTIC_CONTINUE_ON_MISS"
  printf 'track_b_critical_nice=%s\n' "$TRACK_B_CRITICAL_NICE"
  printf 'command=%q ' "$0" "$@"
  printf '\n'
} >"$OUT_DIR/metadata/run_config.txt"

SIGNAL_PID=""
SENDER_PID=""
RECEIVER_PID=""
TRACE_PID=""
GATE_PID=""
cleanup() {
  set +e
  [[ -n "$TRACE_PID" ]] && kill "$TRACE_PID" 2>/dev/null
  [[ -n "$GATE_PID" ]] && kill "$GATE_PID" 2>/dev/null
  [[ -n "$SENDER_PID" ]] && kill "$SENDER_PID" 2>/dev/null
  [[ -n "$RECEIVER_PID" ]] && kill "$RECEIVER_PID" 2>/dev/null
  [[ -n "$SIGNAL_PID" ]] && kill "$SIGNAL_PID" 2>/dev/null
  docker rm -f "$TRACE_CONTAINER" "$SIGNAL_CONTAINER" "$SENDER_CONTAINER" "$RECEIVER_CONTAINER" >/dev/null 2>&1
  docker network rm "$NETWORK" >/dev/null 2>&1
}
trap cleanup EXIT INT TERM

wait_for_log() {
  local path="$1" pattern="$2" attempts=0
  while ! grep -q "$pattern" "$path" 2>/dev/null; do
    attempts=$((attempts + 1))
    if (( attempts > 900 )); then
      echo "Timed out waiting for '$pattern' in $path" >&2
      return 1
    fi
    sleep 0.1
  done
}

docker network create --driver bridge --subnet 172.31.42.0/24 "$NETWORK" \
  >"$OUT_DIR/metadata/docker_network_create.txt"

# Partition the CPUs visible to Docker itself. Each critical validation role
# receives a disjoint cgroup cpuset; the sender container contains only its
# two pacing CPUs plus the four CPUs subsequently assigned to its encoder child.
AVAILABLE_CPUS="$(docker run --rm "$IMAGE" /usr/local/bin/python3 -c \
  'import os; print(",".join(map(str, sorted(os.sched_getaffinity(0)))))')"
"${PYTHON:-python3}" "$ROOT_DIR/scripts/partition_track_b_cpus.py" \
  --cpus "$AVAILABLE_CPUS" >"$OUT_DIR/metadata/cpu_partition.json"
eval "$("${PYTHON:-python3}" "$ROOT_DIR/scripts/partition_track_b_cpus.py" \
  --cpus "$AVAILABLE_CPUS" --shell)"

docker run -d --name "$SENDER_CONTAINER" --privileged \
  --cpuset-cpus "$TRACK_B_SENDER_CONTAINER_CPUS" \
  --network "$NETWORK" --ip 172.31.42.10 \
  -v "$ROOT_DIR:/revo" "$IMAGE" sleep infinity \
  >"$OUT_DIR/metadata/sender_container_id.txt"
docker run -d --name "$RECEIVER_CONTAINER" \
  --cpuset-cpus "$TRACK_B_RECEIVER_CPUS" \
  --network "$NETWORK" --ip 172.31.42.20 \
  -v "$ROOT_DIR:/revo" "$IMAGE" sleep infinity \
  >"$OUT_DIR/metadata/receiver_container_id.txt"
docker run -d --name "$SIGNAL_CONTAINER" \
  --cpuset-cpus "$TRACK_B_SIGNALING_CPUS" \
  --network "$NETWORK" --ip 172.31.42.30 \
  -v "$ROOT_DIR:/revo" "$IMAGE" sleep infinity \
  >"$OUT_DIR/metadata/signaling_container_id.txt"
docker inspect "$SENDER_CONTAINER" "$RECEIVER_CONTAINER" "$SIGNAL_CONTAINER" \
  >"$OUT_DIR/metadata/docker_inspect.json"
docker exec "$SENDER_CONTAINER" ip -brief address \
  >"$OUT_DIR/metadata/sender_interfaces.txt"
docker exec "$RECEIVER_CONTAINER" ip -brief address \
  >"$OUT_DIR/metadata/receiver_interfaces.txt"
docker inspect --format '{{.Name}} cpuset={{.HostConfig.CpusetCpus}}' \
  "$SENDER_CONTAINER" "$RECEIVER_CONTAINER" "$SIGNAL_CONTAINER" \
  >"$OUT_DIR/metadata/effective_container_cpusets.txt"
docker exec "$SENDER_CONTAINER" ip route get 172.31.42.20 \
  >"$OUT_DIR/metadata/sender_route_to_receiver.txt"

# Track B process isolation is a hard precondition, not a best-effort mode.
# Verify it before signaling, receiver, or trace processes are started.
docker exec "$SENDER_CONTAINER" /usr/local/bin/python3 -c \
  'import os,sys; cpus=sorted(os.sched_getaffinity(0)); sender=cpus[:2]; worker=cpus[2:]; print("available_cpus=" + ",".join(map(str,cpus))); print("sender_cpus=" + ",".join(map(str,sender))); print("worker_cpus=" + ",".join(map(str,worker))); sys.exit(0 if len(worker) >= 4 else 2)' \
  >"$OUT_DIR/metadata/encoder_affinity_preflight.txt"
docker exec "$SENDER_CONTAINER" sh -lc \
  "exec nice -n '$TRACK_B_CRITICAL_NICE' /usr/local/bin/python3 -c \
  'import os,sys; value=os.nice(0); print(\"effective_nice=\" + str(value)); sys.exit(0 if value <= int(\"$TRACK_B_CRITICAL_NICE\") else 2)'" \
  >"$OUT_DIR/metadata/scheduler_priority_preflight.txt"

docker exec "$SIGNAL_CONTAINER" sh -lc \
  "cd /revo && exec /usr/local/bin/python3 src/signalling_server.py" \
  >"$OUT_DIR/logs/signaling.log" 2>&1 &
SIGNAL_PID=$!
sleep 1

docker exec "$RECEIVER_CONTAINER" sh -lc \
  "cd /revo && exec /usr/local/bin/python3 src/receiver/receiver-3d.py \
    --out '$CONTAINER_OUT/receiver_rgb.mp4' \
    --out_depth '$CONTAINER_OUT/receiver_depth.mp4' \
    --server_ip 172.31.42.30 --codec h265 \
    --receiver_capacity_csv '$CONTAINER_OUT/receiver_capacity.csv' \
    --diagnostic_csv '$CONTAINER_OUT/receiver_diagnostics.csv' \
    --sctp_event_csv '$CONTAINER_OUT/receiver_sctp_events.csv' \
    --ice_consent_timeout_s 5 --validation_sctp_gap_rtt_fix \
    --receiver_health_window_frames 30 --receiver_decode_guard_s 0.010 \
    --receiver_media_timing_csv '$CONTAINER_OUT/receiver_media_timing.csv' \
    --streaming_validation_output" \
  >"$OUT_DIR/logs/receiver.log" 2>&1 &
RECEIVER_PID=$!
sleep 1

docker exec "$SENDER_CONTAINER" sh -lc \
  "cd /revo && exec nice -n '$TRACK_B_CRITICAL_NICE' /usr/local/bin/python3 src/sender/sender-3d.py \
    --file '$CONTAINER_OUT/input/rgb.mp4' \
    --depth_file '$CONTAINER_OUT/input/depth.mp4' \
    --server_ip 172.31.42.30 --codec h265 \
    --adaptation_mode '$MODE' \
    --controller_csv '$CONTAINER_OUT/controller_decisions.csv' \
    --capacity_probe_csv '$CONTAINER_OUT/capacity_probe.csv' \
    --measurement_csv '$CONTAINER_OUT/sender_measurements.csv' \
    --capacity_csv '$CONTAINER_OUT/sender_capacity_feedback.csv' \
    --diagnostic_csv '$CONTAINER_OUT/sender_diagnostics.csv' \
    --sctp_event_csv '$CONTAINER_OUT/sender_sctp_events.csv' \
    --ice_consent_timeout_s 5 --validation_sctp_gap_rtt_fix \
    --validation_start_signal_path '$CONTAINER_OUT/START' \
    $EVENT_STOP_ARG \
    $FROZEN_TRACK_B_ARGS $DIAGNOSTIC_SENDER_ARGS $TRACK_B_EXTRA_ARGS" \
  >"$OUT_DIR/logs/sender.log" 2>&1 &
SENDER_PID=$!

if [[ -n "$TRACE_PATH" ]]; then
  docker run --rm --name "$TRACE_CONTAINER" --privileged \
    --cpuset-cpus "$TRACK_B_TRACE_CPUS" --network "container:$SENDER_CONTAINER" \
    -v "$ROOT_DIR:/revo" "$IMAGE" sh -lc \
    "cd /revo && exec /usr/local/bin/python3 -u src/sender/run_loss_trace.py \
      --trace '/revo/${TRACE_PATH#"$ROOT_DIR/"}' \
      --interface eth0 \
      --ground_truth_csv '$CONTAINER_OUT/applied_trace.csv' \
      --duration '$RUN_SECONDS' \
      --start_signal_path '$CONTAINER_OUT/START'" \
    >"$OUT_DIR/logs/tc.log" 2>&1 &
  TRACE_PID=$!
  printf '%s\n' "$TRACK_B_TRACE_CPUS" >>"$OUT_DIR/metadata/effective_container_cpusets.txt"
else
  printf 'unshaped no-impairment run\n' >"$OUT_DIR/logs/tc.log"
  printf 'monotonic_timestamp,elapsed_s,trace_time_s,bandwidth_mbps,loss_ratio,delay_ms,interface\n' \
    >"$OUT_DIR/applied_trace.csv"
fi

wait_for_log "$OUT_DIR/logs/sender.log" "waiting for validation start signal"
wait_for_log "$OUT_DIR/logs/receiver.log" "\\[Receiver\\] INIT"
if [[ -n "$TRACE_PID" ]]; then
  wait_for_log "$OUT_DIR/logs/tc.log" "Waiting for validation start signal"
fi
touch "$START_SIGNAL"
if [[ "$EVENT_BASED_LOCAL" == "1" ]]; then
  "${PYTHON:-python3}" "$ROOT_DIR/scripts/track_b_event_gate.py" \
    --run-dir "$OUT_DIR" --stop-signal "$STOP_SIGNAL" \
    --timeout-s "$EVENT_TIMEOUT_S" --gop-size 30 \
    >"$OUT_DIR/logs/event_gate.log" 2>&1 &
  GATE_PID=$!
fi
sleep 3
if [[ -n "$TRACE_PID" ]]; then
  docker exec "$SENDER_CONTAINER" tc -s qdisc show dev eth0 \
    >"$OUT_DIR/metadata/tc_after_3s.txt"
fi

grep -Eq "172\\.31\\.42\\.10.*172\\.31\\.42\\.20.*SUCCEEDED|172\\.31\\.42\\.20.*172\\.31\\.42\\.10.*SUCCEEDED" \
  "$OUT_DIR/logs/sender.log" "$OUT_DIR/logs/receiver.log"
if [[ -n "$TRACE_PID" ]]; then
  test "$(awk 'END {print NR}' "$OUT_DIR/applied_trace.csv")" -gt 1
  grep -q "qdisc" "$OUT_DIR/metadata/tc_after_3s.txt"
fi

if [[ -n "$GATE_PID" ]]; then
  set +e
  wait "$GATE_PID"; GATE_STATUS=$?; GATE_PID=""
  set -e
  printf '%s\n' "$GATE_STATUS" >"$OUT_DIR/metadata/event_gate_exit_status.txt"
  if [[ "$GATE_STATUS" -ne 0 ]]; then
    # Fail closed without killing the docker-exec clients: the sender observes
    # this validation-only stop between frames, closes WebRTC normally, and the
    # receiver then finalizes both streaming MP4 writers in its normal teardown.
    # The gate status remains the authoritative timeout/failure result.
    touch "$STOP_SIGNAL"
  fi
fi

set +e
wait "$SENDER_PID"; SENDER_STATUS=$?; SENDER_PID=""
if [[ "$SENDER_STATUS" -ne 0 ]]; then
  [[ -n "$TRACE_PID" ]] && kill "$TRACE_PID" 2>/dev/null
  kill "$RECEIVER_PID" 2>/dev/null
fi
if [[ -n "$TRACE_PID" ]]; then
  wait "$TRACE_PID"; TRACE_STATUS=$?; TRACE_PID=""
else
  TRACE_STATUS=0
fi
wait "$RECEIVER_PID"; RECEIVER_STATUS=$?; RECEIVER_PID=""
set -e
printf '%s\n' "$TRACE_STATUS" >"$OUT_DIR/metadata/tc_exit_status.txt"
printf '%s\n' "$SENDER_STATUS" >"$OUT_DIR/metadata/sender_exit_status.txt"
printf '%s\n' "$RECEIVER_STATUS" >"$OUT_DIR/metadata/receiver_exit_status.txt"
set +e
"${PYTHON:-python3}" "$ROOT_DIR/scripts/validate_track_b_run.py" \
  --run-dir "$OUT_DIR" --mode "$MODE" --require-media-timing \
  $DIAGNOSTIC_VALIDATOR_ARG
ARTIFACT_STATUS=$?
printf '%s\n' "$ARTIFACT_STATUS" >"$OUT_DIR/metadata/artifact_validator_exit_status.txt"
EVENT_VALIDATION_STATUS=0
if [[ "$EVENT_BASED_LOCAL" == "1" ]]; then
  "${PYTHON:-python3}" "$ROOT_DIR/scripts/track_b_event_gate.py" \
    --run-dir "$OUT_DIR" --validate
  EVENT_VALIDATION_STATUS=$?
  printf '%s\n' "$EVENT_VALIDATION_STATUS" \
    >"$OUT_DIR/metadata/event_gate_validator_exit_status.txt"
fi
set -e
if [[ "${GATE_STATUS:-0}" -ne 0 || "$SENDER_STATUS" -ne 0 || \
      "$RECEIVER_STATUS" -ne 0 || "$TRACE_STATUS" -ne 0 || \
      "$ARTIFACT_STATUS" -ne 0 || "$EVENT_VALIDATION_STATUS" -ne 0 ]]; then
  echo "Track B run failed; see metadata exit-status files" >&2
  exit 1
fi
echo "Completed Track B run: $OUT_DIR"
