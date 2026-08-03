"""
run_loss_trace.py  —  ReVo

Applies time-varying bandwidth and packet loss to a network interface using
Linux tc (traffic control) / netem, replaying a 4-column trace file.

Trace file format (whitespace-separated, one row per time step):
    <timestamp_s>  <bandwidth_mbps>  <rtt_ms>  <loss_0_to_1>

RTT is read from the trace but NOT applied. Delay is fixed at 0 ms so this
player changes only bandwidth and packet loss.

The trace loops automatically when it reaches the end.

Usage (standalone):
    python run_loss_trace.py --trace path/to/trace.log --interface eth0

Normally launched as a subprocess by sender-3d.py via --trace_path.
"""

import time
import subprocess
import sys
import argparse
import os
import re
import signal
import csv

# Advisor requirement: do not add propagation delay or RTT with netem.
FIXED_DELAY_MS = 0


def detect_interface():
    """Return the interface used by the default IPv4 route."""
    result = subprocess.run(
        ["ip", "-o", "route", "show", "default"],
        check=True,
        capture_output=True,
        text=True,
    )
    for line in result.stdout.splitlines():
        fields = line.split()
        if "dev" in fields:
            return fields[fields.index("dev") + 1]
    raise RuntimeError("Could not detect the default network interface")


def parse_args():
    parser = argparse.ArgumentParser(description="ReVo network trace player")
    parser.add_argument("--interface", default=None, help="Network interface (default: detect from IPv4 route)")
    parser.add_argument("--trace",     required=True,      help="Path to the trace file")
    parser.add_argument("--ground_truth_csv", default=None,
                        help="Optional CSV path for monotonic timestamps of applied tc updates")
    parser.add_argument("--duration", type=float, default=None,
                        help="Optional max replay duration in seconds")
    parser.add_argument("--loss_override", type=float, default=None,
                        help="Optional loss fraction override, e.g. 0.0 for bandwidth-only validation")
    parser.add_argument("--delay_override_ms", type=float, default=None,
                        help="Optional validation-only netem delay override in milliseconds")
    parser.add_argument("--start_signal_path", default="",
                        help="Validation only: wait for this file after tc setup before replay")
    return parser.parse_args()


def _validate_interface(interface: str):
    if not re.fullmatch(r"[A-Za-z0-9_.:-]+", interface):
        raise ValueError(f"Invalid interface name: {interface!r}")
    if not os.path.exists(f"/sys/class/net/{interface}"):
        raise ValueError(f"Network interface does not exist: {interface}")


def _run_tc(*args: str, ignore_errors=False):
    """Run tc through passwordless sudo and surface unexpected failures."""
    prefix = [] if os.geteuid() == 0 else ["sudo", "-n"]
    result = subprocess.run(
        [*prefix, "tc", *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode and not ignore_errors:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"tc {' '.join(args)} failed: {detail}")
    return result


def setup_tc(interface: str):
    """
    Build the tc hierarchy on the interface:
      root HTB qdisc → HTB class (rate updated per step) → netem child (loss only)
    Any existing root qdisc is deleted first.
    """
    print(f"[tc] Setting up on {interface}")
    _run_tc("qdisc", "del", "dev", interface, "root", ignore_errors=True)
    _run_tc("qdisc", "add", "dev", interface, "root", "handle", "1:", "htb", "default", "10")
    _run_tc("class", "add", "dev", interface, "parent", "1:", "classid", "1:10", "htb", "rate", "1000Mbit")
    _run_tc(
        "qdisc", "add", "dev", interface, "parent", "1:10", "handle", "10:",
        "netem", "delay", "0ms", "loss", "0%",
    )


def update_tc(interface: str, bandwidth_mbps: float, loss_ratio: float,
              delay_ms=FIXED_DELAY_MS):
    """
    Apply one trace step: update bandwidth (HTB class) and loss (netem).
    loss_ratio is in [0, 1]; converted to percent for netem.
    """
    if bandwidth_mbps <= 0:
        raise ValueError(f"Bandwidth must be positive, got {bandwidth_mbps}")
    loss_pct = min(100.0, max(0.0, loss_ratio * 100.0))
    _run_tc(
        "class", "change", "dev", interface, "parent", "1:", "classid", "1:10",
        "htb", "rate", f"{bandwidth_mbps}Mbit", "burst", "150k",
    )
    _run_tc(
        "qdisc", "change", "dev", interface, "parent", "1:10", "handle", "10:",
        "netem", "delay", f"{delay_ms}ms", "loss", f"{loss_pct}%",
    )
    sys.stdout.write(
        f"\r[trace] BW: {bandwidth_mbps:.1f} Mbps | "
        f"Loss: {loss_pct:.1f}% | Delay: {delay_ms:g} ms | t={time.perf_counter():.2f}s"
    )
    sys.stdout.flush()
    return time.perf_counter(), loss_pct, delay_ms


def cleanup_tc(interface: str):
    """Remove all tc rules from the interface."""
    print(f"\n[tc] Cleaning up on {interface}")
    _run_tc("qdisc", "del", "dev", interface, "root", ignore_errors=True)


def _handle_stop_signal(signum, _frame):
    raise KeyboardInterrupt(f"received signal {signum}")


def main():
    args = parse_args()
    interface = args.interface or detect_interface()
    _validate_interface(interface)

    if not os.path.exists(args.trace):
        raise FileNotFoundError(f"Trace file not found: {args.trace}")

    signal.signal(signal.SIGTERM, _handle_stop_signal)
    signal.signal(signal.SIGHUP, _handle_stop_signal)

    gt_file = None
    gt_writer = None

    try:
        setup_tc(interface)

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

        with open(args.trace, "r") as f:
            lines = [ln.strip() for ln in f if ln.strip() and not ln.lstrip().startswith("#")]

        if not lines:
            raise ValueError("Trace file is empty")

        if args.start_signal_path:
            print(f"[tc] Waiting for validation start signal: {args.start_signal_path}")
            while not os.path.exists(args.start_signal_path):
                time.sleep(0.01)
            print("[tc] Validation start signal observed")

        # Total duration of one pass — used to offset timestamps on loop
        trace_duration = float(lines[-1].split()[0])
        num_lines      = len(lines)
        line_idx       = 0
        loop_offset    = 0.0
        start_time     = time.perf_counter()

        while True:
            elapsed = time.perf_counter() - start_time
            if args.duration is not None and elapsed >= args.duration:
                break

            current_idx = line_idx % num_lines

            # Advance the time offset each time the trace wraps around
            if line_idx > 0 and current_idx == 0:
                loop_offset += trace_duration

            parts = lines[current_idx].split()
            if len(parts) < 4:
                line_idx += 1
                continue

            # Columns: timestamp  bandwidth_mbps  rtt_ms  loss_fraction
            target_time    = float(parts[0]) + loop_offset
            bandwidth_mbps = float(parts[1])
            # parts[2] is rtt_ms — read but not applied (delay is fixed at zero)
            loss_ratio     = max(0.0, float(parts[3]))
            if args.loss_override is not None:
                loss_ratio = max(0.0, float(args.loss_override))

            # Wait until wall time matches the trace timestamp
            sleep_s = target_time - (time.perf_counter() - start_time)
            if sleep_s > 0:
                time.sleep(sleep_s)

            delay_ms = (
                FIXED_DELAY_MS
                if args.delay_override_ms is None
                else float(args.delay_override_ms)
            )
            applied_ts, _loss_pct, applied_delay_ms = update_tc(
                interface, bandwidth_mbps, loss_ratio, delay_ms
            )
            if gt_writer is not None:
                gt_writer.writerow({
                    "monotonic_timestamp": f"{applied_ts:.9f}",
                    "elapsed_s": f"{applied_ts - start_time:.9f}",
                    "trace_time_s": f"{target_time:.9f}",
                    "bandwidth_mbps": f"{bandwidth_mbps:.9f}",
                    "loss_ratio": f"{loss_ratio:.9f}",
                    "delay_ms": applied_delay_ms,
                    "interface": interface,
                })
                gt_file.flush()
            line_idx += 1

    except KeyboardInterrupt:
        print("\n[tc] Interrupted.")
    except Exception as e:
        print(f"\n[tc] Error: {e}")
        raise
    finally:
        if gt_file is not None:
            gt_file.close()
        cleanup_tc(interface)


if __name__ == "__main__":
    main()
