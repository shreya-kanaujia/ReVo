#!/usr/bin/env python3
"""Sequential, resumable local Wi-Fi evaluation runner.

A run is publishable only after process exits, tc evidence, logs, videos, source
window, and corruption mask have all been validated.  Resume is driven solely
by that validated per-run status document.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
COMPOSE = ROOT / "compose.local-testbed.yml"
TRACE_MAP = ROOT / "src/sender/trace_map.txt"
SMOKE_TIMEOUT_S = 180.0
FULL_TEARDOWN_MARGIN_S = 180.0
RUNTIME_IDENTITY_FILES = {
    "REVO_EXPECTED_RECEIVER_SHA256": ROOT / "src/receiver/receiver-3d.py",
    "REVO_EXPECTED_SENDER_SHA256": ROOT / "src/sender/sender-3d.py",
    "REVO_EXPECTED_TC_SHA256": ROOT / "scripts/capture_tc_stats.sh",
    "REVO_EXPECTED_GUARD_SHA256": ROOT / "scripts/verify_wifi_eval_runtime.py",
    "REVO_EXPECTED_GCC_SHA256": ROOT / "src/gcc_controller.py",
    "REVO_EXPECTED_GCC_CODEC_SHA256": ROOT / "src/sender/H265_wrapper_v2.py",
}

SHARED_CONFIG = {
    "SALSIFY_DUP_DELAY_MS": "33",
    "SALSIFY_RGB_QP_HI": "25",
    "SALSIFY_RGB_QP_MID": "30",
    "SALSIFY_RGB_QP_LO": "35",
    "SALSIFY_DEPTH_QP_HI": "25",
    "SALSIFY_DEPTH_QP_MID": "30",
    "SALSIFY_DEPTH_QP_LO": "35",
    "SALSIFY_START": "1",
    "SALSIFY_SOFT_FRAC": "0.5",
    "SALSIFY_MISS_THRESH": "8.0",
    "SALSIFY_MISS_CLEAR": "4.0",
    "SALSIFY_MIN_DWELL": "2",
    "SALSIFY_GOP": "30",
    "SALSIFY_GOP_TIERS": "",
    "SALSIFY_KF_COOLDOWN": "10",
    "SALSIFY_KF_SKIP_NEAR": "10",
    "SALSIFY_FEC_ADAPT": "1",
    "SALSIFY_FEC_MIN": "0.5",
    "SALSIFY_FEC_MAX": "1.5",
    "SALSIFY_FEC_MAXPK": "24",
    "SALSIFY_FEC_COMPENSATE": "0",
    "REVO_GOP": "30",
    "REVO_FB_INTERVAL": "5",
    "REVO_FB_WINDOW": "30",
    "REVO_KF_RETRY_S": "0.25",
    "REVO_PLAYOUT_MS": "1000",
    "REVO_FIX_CLOCK": "0",
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def get_git_commit():
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
        text=True, check=True
    ).stdout.strip()


def parse_trace_map(path=TRACE_MAP):
    mappings = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 3 and parts[0] == "wifi":
                stem = parts[1]
                trace_basename = os.path.basename(parts[2])
                mappings.append({
                    "video_stem": stem,
                    "trace_basename": trace_basename,
                    "rgb_source": f"data/gt_rgb/{stem}.mp4",
                    "depth_source": f"data/gt_depth/{stem}_vis.mp4",
                    "trace_path": f"src/sender/traces/wifi/{trace_basename}",
                })
    if len(mappings) != 30 or len({m["video_stem"] for m in mappings}) != 30:
        raise RuntimeError(f"expected 30 unique Wi-Fi mappings, found {len(mappings)}")
    return mappings


def video_info(path):
    proc = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-count_frames", "-show_entries",
        "stream=avg_frame_rate,nb_frames,nb_read_frames,duration",
        "-of", "json", str(path),
    ], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {proc.stderr.strip()}")
    streams = json.loads(proc.stdout).get("streams", [])
    if not streams:
        raise RuntimeError(f"no readable video stream in {path}")
    stream = streams[0]
    num, den = stream["avg_frame_rate"].split("/")
    frames = stream.get("nb_read_frames") or stream.get("nb_frames")
    return {
        "fps": float(num) / float(den),
        "frames": int(frames),
        "duration_s": float(stream.get("duration", 0.0)),
    }


def method_config(method):
    config = dict(SHARED_CONFIG)
    config["SALSIFY_MODE"] = "1" if method == "abr" else "0"
    config["REVO_GCC_MODE"] = "1" if method == "gcc" else "0"
    # Mode 0 constructs the persistent baseline encoders at MID; spelling out
    # the complete ladder still freezes the comparison rather than relying on
    # runtime defaults.
    return config


def runtime_config(method, smoke, limit_frames, source_frames):
    config = method_config(method)
    config["REVO_SOURCE_FRAMES"] = str(source_frames)
    config["REVO_EVAL_FULL_RUN"] = "0" if smoke else "1"
    config.update({key: sha256_file(path)
                   for key, path in RUNTIME_IDENTITY_FILES.items()})
    if smoke:
        config["REVO_MAX_FRAMES"] = str(limit_frames)
    else:
        config.pop("REVO_MAX_FRAMES", None)
    return config


def verify_runtime_container(env):
    """Prove Compose sees the current bind-mounted sources before networking."""
    for role in ("sender", "receiver"):
        proc = subprocess.run([
            "docker", "compose", "-f", str(COMPOSE), "run", "--rm",
            "--no-deps", "--entrypoint", "python", role,
            "scripts/verify_wifi_eval_runtime.py", "--role", role,
        ], cwd=ROOT, env=env, capture_output=True, text=True)
        if proc.returncode != 0 or "RUNTIME_IDENTITY_OK" not in proc.stdout:
            raise RuntimeError(
                f"runtime preflight failed for {role}: "
                f"stdout={proc.stdout.strip()} stderr={proc.stderr.strip()}"
            )
        print(proc.stdout.strip())


def build_run_environment(method, smoke, limit_frames, source_frames, mapping,
                          base_environment=None):
    env = dict(os.environ if base_environment is None else base_environment)
    # A caller's old smoke variable must never leak into a full evaluation.
    if not smoke:
        env.pop("REVO_MAX_FRAMES", None)
    env.update(runtime_config(method, smoke, limit_frames, source_frames))
    env.update({
        "RGB_SOURCE": mapping["rgb_source"],
        "DEPTH_SOURCE": mapping["depth_source"],
        "TRACE_PATH": mapping["trace_path"],
    })
    return env


def timeout_for_run(smoke, source_duration_s):
    if smoke:
        return SMOKE_TIMEOUT_S
    return source_duration_s + FULL_TEARDOWN_MARGIN_S


def _write_immutable_json(path, content):
    serialized = json.dumps(content, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text() != serialized:
            raise RuntimeError(f"frozen manifest mismatch: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialized)


def ensure_manifest(output_root, mappings):
    manifest_dir = output_root / "manifest"
    enriched = []
    for mapping in mappings:
        row = dict(mapping)
        rgb = ROOT / row["rgb_source"]
        depth = ROOT / row["depth_source"]
        trace = ROOT / row["trace_path"]
        if not (rgb.exists() and depth.exists() and trace.exists()):
            raise FileNotFoundError(f"missing mapped asset for {row['video_stem']}")
        rgb_info, depth_info = video_info(rgb), video_info(depth)
        if abs(rgb_info["fps"] - depth_info["fps"]) > 1e-6:
            raise RuntimeError(f"RGB/depth FPS mismatch for {row['video_stem']}")
        row.update({
            "rgb_sha256": sha256_file(rgb),
            "depth_sha256": sha256_file(depth),
            "trace_sha256": sha256_file(trace),
            "source_fps": rgb_info["fps"],
            "rgb_frames": rgb_info["frames"],
            "depth_frames": depth_info["frames"],
        })
        enriched.append(row)

    trace_player = ROOT / "src/sender/run_loss_trace.py"
    manifest = {
        "schema_version": 1,
        "git_commit": get_git_commit(),
        "wifi_mapping_count": len(enriched),
        "mappings": enriched,
        "run_loss_trace": {
            "path": "src/sender/run_loss_trace.py",
            "sha256": sha256_file(trace_player),
            "semantics": {
                "interface": "eth0",
                "fixed_one_way_delay_ms": 40,
                "trace_rtt_column_applied": False,
                "trace_loops": True,
                "bandwidth_column": "decimal Mbps",
                "loss_column": "fraction converted to percent",
            },
        },
        "cdf_population": (
            "frame-level pooling of source frame IDs whose recovery-aware mask "
            "is nonzero; raw and recovered scores use the identical frame IDs"
        ),
    }
    _write_immutable_json(manifest_dir / "experiment_manifest.json", manifest)
    for method in ("baseline", "abr", "gcc"):
        _write_immutable_json(
            manifest_dir / f"{method}_config.json",
            {"method": method, "configuration": method_config(method)},
        )
    return manifest


def container_state(name):
    proc = subprocess.run(
        ["docker", "inspect", name], capture_output=True, text=True
    )
    if proc.returncode != 0:
        return {"exists": False, "running": False, "exit_code": None}
    state = json.loads(proc.stdout)[0]["State"]
    return {
        "exists": True,
        "running": bool(state["Running"]),
        "exit_code": state.get("ExitCode"),
        "error": state.get("Error", ""),
        "finished_at": state.get("FinishedAt", ""),
    }


def wait_for_exit(names, timeout_s):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        states = {name: container_state(name) for name in names}
        if all(state["exists"] and not state["running"] for state in states.values()):
            return states, False
        time.sleep(1.0)
    return {name: container_state(name) for name in names}, True


def first_successful_fid(log_text):
    match = re.search(r"Successfully decoded keyframe\s+(\d+)", log_text)
    return int(match.group(1)) if match else 0


def validate_tc_evidence(path):
    text = Path(path).read_text() if Path(path).exists() else ""
    required = ["capture_exit_status=0", "qdisc htb 1:", "qdisc netem 10:", "Sent "]
    missing = [item for item in required if item not in text]
    netem = re.search(r"qdisc netem 10:.*?(?=^qdisc |^class_exit_status=|\Z)",
                      text, flags=re.MULTILINE | re.DOTALL)
    sent = (re.search(r"Sent\s+(\d+)\s+bytes\s+(\d+)\s+pkt.*?dropped\s+(\d+)",
                      netem.group(0), flags=re.DOTALL) if netem else None)
    if missing or sent is None or int(sent.group(1)) <= 0 or int(sent.group(2)) <= 0:
        raise RuntimeError(f"invalid netem evidence {path}; missing={missing}")
    return {
        "bytes": int(sent.group(1)), "packets": int(sent.group(2)),
        "drops": int(sent.group(3)),
    }


def validate_run_artifacts(paths, source_frames, source_fps):
    for key in ("sender_log", "receiver_log", "rgb", "depth", "tc"):
        if not paths[key].exists() or paths[key].stat().st_size == 0:
            raise RuntimeError(f"missing or empty {key}: {paths[key]}")
    sender_log = paths["sender_log"].read_text(errors="replace")
    receiver_log = paths["receiver_log"].read_text(errors="replace")
    if "[Sender] Completed streaming all frames." not in sender_log:
        raise RuntimeError("sender completion marker absent")
    if "[Receiver] Graceful shutdown complete" not in receiver_log:
        raise RuntimeError("receiver graceful-shutdown marker absent")
    if "Traceback (most recent call last)" in sender_log or "Traceback (most recent call last)" in receiver_log:
        raise RuntimeError("traceback present in sender/receiver log")
    if f"[Receiver] Source window classified: 0..{source_frames - 1}" not in receiver_log:
        raise RuntimeError("receiver did not classify the complete source window")

    fid0 = first_successful_fid(receiver_log)
    expected_output_frames = source_frames - fid0
    rgb_info, depth_info = video_info(paths["rgb"]), video_info(paths["depth"])
    for label, info in (("RGB", rgb_info), ("depth", depth_info)):
        if abs(info["fps"] - source_fps) > 1e-6:
            raise RuntimeError(f"{label} FPS {info['fps']} != source FPS {source_fps}")
        if info["frames"] != expected_output_frames:
            raise RuntimeError(
                f"{label} frames {info['frames']} != expected {expected_output_frames} "
                f"for fid0={fid0}, source_frames={source_frames}"
            )
    tc = validate_tc_evidence(paths["tc"])
    return {"fid0": fid0, "expected_output_frames": expected_output_frames,
            "rgb": rgb_info, "depth": depth_info, "tc": tc}


def success_marker_valid(path):
    try:
        status = json.loads(Path(path).read_text())
        if status.get("status") != "VALIDATED_COMPLETE":
            return False
        for artifact in status["artifacts"].values():
            if not Path(artifact).exists() or Path(artifact).stat().st_size == 0:
                return False
        rgb = video_info(status["artifacts"]["rgb"])
        depth = video_info(status["artifacts"]["depth"])
        validate_tc_evidence(status["artifacts"]["tc"])
        return (rgb["frames"] == status["validation"]["expected_output_frames"]
                and depth["frames"] == status["validation"]["expected_output_frames"])
    except Exception:
        return False


def paths_for(method_dir, stem, temporary=False):
    if temporary:
        return {
            "rgb": ROOT / "output/temp_receiver_rgb.mp4",
            "depth": ROOT / "output/temp_receiver_depth.mp4",
            "receiver_log": ROOT / "output/temp_receiver.log",
            "sender_log": ROOT / "output/temp_sender.log",
            "signaling_log": ROOT / "output/temp_signaling.log",
            "tc": ROOT / "output/temp_tc_stats.log",
        }
    return {
        "rgb": method_dir / "receiver/rgb" / f"{stem}.mp4",
        "depth": method_dir / "receiver/depth" / f"{stem}_vis.mp4",
        "receiver_log": method_dir / "receiver/logs" / f"{stem}.log",
        "sender_log": method_dir / "sender_logs" / f"{stem}.log",
        "signaling_log": method_dir / "signaling_logs" / f"{stem}.log",
        "tc": method_dir / "metrics" / f"{stem}_tc_stats.log",
        "mask": method_dir / "receiver/masks" / f"{stem}_frame_mask.npy",
        "status": method_dir / "status" / f"{stem}.json",
    }


def run_one(method, mapping, smoke, limit_frames, output_root):
    method_dir = output_root / method
    final = paths_for(method_dir, mapping["video_stem"])
    if success_marker_valid(final["status"]):
        print(f"Skipping validated run: {mapping['video_stem']}")
        return "SKIPPED_VALIDATED"

    rgb_source = ROOT / mapping["rgb_source"]
    depth_source = ROOT / mapping["depth_source"]
    rgb_info, depth_info = video_info(rgb_source), video_info(depth_source)
    if abs(rgb_info["fps"] - depth_info["fps"]) > 1e-6:
        raise RuntimeError("source RGB/depth FPS mismatch")
    source_frames = min(rgb_info["frames"], depth_info["frames"])
    if smoke:
        source_frames = min(source_frames, limit_frames)
    env = build_run_environment(
        method, smoke, limit_frames, source_frames, mapping)

    subprocess.run(["docker", "compose", "-f", str(COMPOSE), "down"], cwd=ROOT, check=True)
    verify_runtime_container(env)
    temporary = paths_for(method_dir, mapping["video_stem"], temporary=True)
    for path in temporary.values():
        if path.exists():
            path.unlink()

    process_status = {}
    timed_out = False
    try:
        subprocess.run(
            ["docker", "compose", "-f", str(COMPOSE), "up", "-d"],
            cwd=ROOT, env=env, check=True,
        )
        timeout = timeout_for_run(smoke, source_frames / rgb_info["fps"])
        states, timed_out = wait_for_exit(["revo-sender", "revo-receiver"], timeout)
        process_status.update(states)
        if timed_out:
            raise RuntimeError(f"run timeout after {timeout:.1f}s")

        signaling_before = container_state("revo-signaling")
        process_status["revo-signaling_before_stop"] = signaling_before
        if not signaling_before["running"]:
            raise RuntimeError("signaling server exited before teardown")
        subprocess.run(
            ["docker", "stop", "--signal", "SIGINT", "--time", "10", "revo-signaling"],
            cwd=ROOT, check=True, capture_output=True, text=True,
        )
        process_status["revo-signaling"] = container_state("revo-signaling")

        for service in ("revo-sender", "revo-receiver", "revo-signaling"):
            if process_status[service]["exit_code"] != 0:
                raise RuntimeError(f"{service} exit={process_status[service]['exit_code']}")

        validation = validate_run_artifacts(temporary, source_frames, rgb_info["fps"])

        sys.path.insert(0, str(ROOT / "scripts"))
        from generate_frame_masks import parse_log_file
        _, mask = parse_log_file(temporary["receiver_log"], max_frames=source_frames)
        if len(mask) != source_frames:
            raise RuntimeError(f"mask length {len(mask)} != source window {source_frames}")

        for path in final.values():
            path.parent.mkdir(parents=True, exist_ok=True)
        for key in ("rgb", "depth", "receiver_log", "sender_log", "signaling_log", "tc"):
            os.replace(temporary[key], final[key])
        np.save(final["mask"], mask)

        status = {
            "status": "VALIDATED_COMPLETE",
            "method": method,
            "video_stem": mapping["video_stem"],
            "mapping": mapping,
            "source_frames": source_frames,
            "source_fps": rgb_info["fps"],
            "smoke": smoke,
            "process_status": process_status,
            "validation": validation,
            "mask": {
                "length": len(mask),
                "type_0": int(np.count_nonzero(mask == 0)),
                "type_1": int(np.count_nonzero(mask == 1)),
                "type_2": int(np.count_nonzero(mask == 2)),
            },
            "artifacts": {key: str(path) for key, path in final.items()
                          if key not in ("status",)},
        }
        final["status"].write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
        print(f"VALIDATED_COMPLETE: {mapping['video_stem']} ({method})")
        return "VALIDATED_COMPLETE"
    finally:
        subprocess.run(
            ["docker", "compose", "-f", str(COMPOSE), "down"],
            cwd=ROOT, check=False, capture_output=True,
        )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="ReVo Wi-Fi evaluation batch runner")
    parser.add_argument("--method", choices=["baseline", "abr", "gcc"], required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--full_run", action="store_true")
    parser.add_argument("--smoke_stem", default="1lSejjfNHpw_0075_S0_E728_L671_T47_R1471_B847")
    parser.add_argument("--limit_frames", type=int, default=150)
    parser.add_argument("--output_root", default="output/wifi_30video_eval")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.chdir(ROOT)
    mappings = parse_trace_map()
    output_root = ROOT / args.output_root
    ensure_manifest(output_root, mappings)
    selected = mappings
    if args.smoke:
        selected = [m for m in mappings if m["video_stem"] == args.smoke_stem]
        if not selected:
            raise RuntimeError(f"smoke stem not in fixed Wi-Fi mapping: {args.smoke_stem}")
    for index, mapping in enumerate(selected, 1):
        print(f"[{index}/{len(selected)}] {args.method}: {mapping['video_stem']}")
        run_one(args.method, mapping, args.smoke, args.limit_frames, output_root)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
