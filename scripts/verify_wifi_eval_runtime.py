#!/usr/bin/env python3
"""Fail-fast identity/config guard executed inside Wi-Fi runtime containers."""

import argparse
import hashlib
import os
from pathlib import Path
import sys


FILES = {
    "receiver": "src/receiver/receiver-3d.py",
    "sender": "src/sender/sender-3d.py",
    "tc": "scripts/capture_tc_stats.sh",
    "guard": "scripts/verify_wifi_eval_runtime.py",
    "gcc": "src/gcc_controller.py",
    "gcc_codec": "src/sender/H265_wrapper_v2.py",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(role):
    failures = []
    for label, path in FILES.items():
        expected = os.environ.get(f"REVO_EXPECTED_{label.upper()}_SHA256", "")
        actual = digest(path) if Path(path).is_file() else "MISSING"
        if not expected or actual != expected:
            failures.append(f"{label}: expected={expected or 'UNSET'} actual={actual}")
    if role == "sender" and os.environ.get("SALSIFY_DUP_DELAY_MS") != "33":
        failures.append("SALSIFY_DUP_DELAY_MS must be 33")
    gcc_mode = os.environ.get("REVO_GCC_MODE", "0")
    if gcc_mode not in ("0", "1"):
        failures.append(f"REVO_GCC_MODE must be 0 or 1, got {gcc_mode}")
    if role == "sender" and gcc_mode == "1" and os.environ.get("SALSIFY_MODE") != "0":
        failures.append("GCC must be isolated from the ABR controller")
    if role == "sender":
        for variable in ("RGB_SOURCE", "DEPTH_SOURCE", "TRACE_PATH"):
            value = os.environ.get(variable, "")
            if not value or not Path(value).is_file():
                failures.append(f"{variable} does not resolve: {value or 'UNSET'}")
    if os.environ.get("REVO_EVAL_FULL_RUN") == "1" and os.environ.get("REVO_MAX_FRAMES"):
        failures.append("full_run must not set REVO_MAX_FRAMES")
    receiver_text = Path(FILES["receiver"]).read_text()
    if 'WIFI_EVAL_SOURCE_WINDOW_VERSION = "2"' not in receiver_text:
        failures.append("receiver source-window contract v2 absent")
    tc_text = Path(FILES["tc"]).read_text()
    if 'qdisc netem 10:' not in tc_text or 'netem_snapshot' not in tc_text:
        failures.append("tc netem-child validation contract absent")
    if failures:
        print("RUNTIME_IDENTITY_MISMATCH: " + "; ".join(failures), file=sys.stderr)
        return 1
    print(
        f"RUNTIME_IDENTITY_OK role={role} full_run={os.environ.get('REVO_EVAL_FULL_RUN')} "
        f"max_frames={os.environ.get('REVO_MAX_FRAMES', '<unset>')} "
        f"duplicate_ms={os.environ.get('SALSIFY_DUP_DELAY_MS', '<not-applicable>')} "
        f"gcc_mode={gcc_mode} "
        f"rgb={os.environ.get('RGB_SOURCE', '<not-applicable>')} "
        f"depth={os.environ.get('DEPTH_SOURCE', '<not-applicable>')} "
        f"trace={os.environ.get('TRACE_PATH', '<not-applicable>')}"
    )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=["sender", "receiver"], required=True)
    args = parser.parse_args()
    raise SystemExit(verify(args.role))
