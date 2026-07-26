"""
run_loss_trace.py  —  ReVo

Applies time-varying bandwidth and packet loss to a network interface using
Linux tc (traffic control) / netem, replaying a 4-column trace file.

Trace file format (whitespace-separated, one row per time step):
    <timestamp_s>  <bandwidth_mbps>  <rtt_ms>  <loss_0_to_1>

RTT is read from the trace but NOT applied — delay is fixed at FIXED_DELAY_MS.
This is intentional: in our lab setup the propagation delay is controlled
separately, so only bandwidth and loss are emulated here.

The trace loops automatically when it reaches the end.

Usage (standalone):
    sudo python run_loss_trace.py --trace path/to/trace.log --interface eth0

Normally launched as a subprocess by sender-3d.py via --trace_path.
"""

import time
import subprocess
import sys
import argparse
import csv
import os

# One-way delay applied to all packets. RTT column in the trace is ignored.
FIXED_DELAY_MS = 40


def parse_args():
    parser = argparse.ArgumentParser(description="ReVo network trace player")
    parser.add_argument("--interface", default="enp130s0", help="Network interface (e.g. eth0)")
    parser.add_argument("--trace", required=True, help="Path to the trace file")
    parser.add_argument("--ground_truth_csv", default=None,
                        help="Optional CSV path for monotonic timestamps of applied tc updates")
    parser.add_argument("--duration", type=float, default=None,
                        help="Optional max replay duration in seconds")
    parser.add_argument("--loss_override", type=float, default=None,
                        help="Optional validation-only loss fraction override")
    parser.add_argument("--delay_override_ms", type=float, default=None,
                        help="Optional validation-only netem delay override in milliseconds")
    return parser.parse_args()


def _run(cmd: str):
    """Run a shell command silently; ignore errors (tc may warn on first delete)."""
    subprocess.run(cmd, shell=True, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def setup_tc(interface: str):
    """
    Build the tc hierarchy on the interface:
      root HTB qdisc → HTB class (rate updated per step) → netem child (loss + delay)
    Any existing root qdisc is deleted first.
    """
    print(f"[tc] Setting up on {interface}")
    _run(f"tc qdisc del dev {interface} root")
    _run(f"tc qdisc  add dev {interface} root handle 1: htb default 10")
    _run(f"tc class  add dev {interface} parent 1: classid 1:10 htb rate 1000Mbit")
    _run(f"tc qdisc  add dev {interface} parent 1:10 handle 10: netem delay 0ms loss 0%")


def update_tc(interface: str, bandwidth_mbps: float, loss_ratio: float,
              delay_ms=FIXED_DELAY_MS):
    """Apply one trace step and return its absolute monotonic timestamp."""
    loss_pct = max(0.0, loss_ratio * 100.0)
    _run(f"tc class change dev {interface} parent 1: classid 1:10 htb rate {bandwidth_mbps}Mbit burst 150k")
    _run(f"tc qdisc change dev {interface} parent 1:10 handle 10: netem delay {delay_ms}ms loss {loss_pct}%")
    applied_ts = time.perf_counter()
    sys.stdout.write(
        f"\r[trace] BW: {bandwidth_mbps:.1f} Mbps | Loss: {loss_pct:.1f}% | "
        f"Delay: {delay_ms:g} ms | t={applied_ts:.2f}s"
    )
    sys.stdout.flush()
    return applied_ts, loss_pct, delay_ms


def cleanup_tc(interface: str):
    """Remove all tc rules from the interface."""
    print(f"\n[tc] Cleaning up on {interface}")
    _run(f"tc qdisc del dev {interface} root")


def main():
    args = parse_args()

    if not os.path.exists(args.trace):
        print(f"Error: trace file not found: {args.trace}")
        return

    setup_tc(args.interface)

    with open(args.trace, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    if not lines:
        print("Error: trace file is empty.")
        cleanup_tc(args.interface)
        return

    trace_duration = float(lines[-1].split()[0])
    num_lines = len(lines)
    line_idx = 0
    loop_offset = 0.0
    start_time = time.perf_counter()
    gt_file = None
    gt_writer = None

    try:
        if args.ground_truth_csv:
            os.makedirs(os.path.dirname(os.path.abspath(args.ground_truth_csv)), exist_ok=True)
            gt_file = open(args.ground_truth_csv, "w", newline="")
            gt_writer = csv.DictWriter(
                gt_file,
                fieldnames=[
                    "monotonic_timestamp",
                    "elapsed_s",
                    "trace_time_s",
                    "bandwidth_mbps",
                    "loss_ratio",
                    "delay_ms",
                    "interface",
                ],
            )
            gt_writer.writeheader()

        while True:
            elapsed = time.perf_counter() - start_time
            if args.duration is not None and elapsed >= args.duration:
                break

            current_idx = line_idx % num_lines
            if line_idx > 0 and current_idx == 0:
                loop_offset += trace_duration

            parts = lines[current_idx].split()
            if len(parts) < 4:
                line_idx += 1
                continue

            target_time = float(parts[0]) + loop_offset
            bandwidth_mbps = float(parts[1])
            loss_ratio = max(0.0, float(parts[3]))
            if args.loss_override is not None:
                loss_ratio = max(0.0, float(args.loss_override))
            delay_ms = (
                FIXED_DELAY_MS
                if args.delay_override_ms is None
                else float(args.delay_override_ms)
            )

            sleep_s = target_time - (time.perf_counter() - start_time)
            if sleep_s > 0:
                time.sleep(sleep_s)

            applied_ts, _loss_pct, applied_delay_ms = update_tc(
                args.interface, bandwidth_mbps, loss_ratio, delay_ms
            )
            if gt_writer is not None:
                gt_writer.writerow({
                    "monotonic_timestamp": f"{applied_ts:.9f}",
                    "elapsed_s": f"{applied_ts - start_time:.9f}",
                    "trace_time_s": f"{target_time:.9f}",
                    "bandwidth_mbps": f"{bandwidth_mbps:.9f}",
                    "loss_ratio": f"{loss_ratio:.9f}",
                    "delay_ms": applied_delay_ms,
                    "interface": args.interface,
                })
                gt_file.flush()
            line_idx += 1

    except KeyboardInterrupt:
        print("\n[tc] Interrupted.")
    except Exception as e:
        print(f"\n[tc] Error: {e}")
    finally:
        if gt_file is not None:
            gt_file.close()
        cleanup_tc(args.interface)


if __name__ == "__main__":
    main()
