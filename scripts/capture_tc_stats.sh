#!/usr/bin/env bash

set -u

interface="${1:-eth0}"
output_path="${2:-output/tc_stats.log}"
attempts="${TC_CAPTURE_ATTEMPTS:-200}"
interval_s="${TC_CAPTURE_INTERVAL_S:-0.05}"

mkdir -p "$(dirname "$output_path")"

for ((attempt = 1; attempt <= attempts; attempt++)); do
    qdisc_snapshot="$(tc -s qdisc show dev "$interface" 2>&1)"
    qdisc_status=$?
    if [[ $qdisc_status -eq 0 && "$qdisc_snapshot" == *"netem"* ]] \
        && grep -Eq 'Sent [1-9][0-9]* bytes' <<< "$qdisc_snapshot" \
        && grep -Eq 'dropped [1-9][0-9]*' <<< "$qdisc_snapshot"; then
        class_snapshot="$(tc -s class show dev "$interface" 2>&1)"
        class_status=$?
        {
            printf 'timestamp_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
            printf 'interface=%s\n' "$interface"
            printf 'qdisc_exit_status=%d\n' "$qdisc_status"
            printf '%s\n' '--- tc -s qdisc ---'
            printf '%s\n' "$qdisc_snapshot"
            printf 'class_exit_status=%d\n' "$class_status"
            printf '%s\n' '--- tc -s class ---'
            printf '%s\n' "$class_snapshot"
        } > "$output_path"
        exit "$class_status"
    fi
    sleep "$interval_s"
done

{
    printf 'timestamp_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'interface=%s\n' "$interface"
    printf 'capture_exit_status=1\n'
    printf 'error=netem qdisc was not observed within the bounded capture window\n'
} > "$output_path"
exit 1
