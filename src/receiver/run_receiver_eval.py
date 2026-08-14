"""
run_receiver_eval.py  —  ReVo

Persistent receiver service for batch evaluation.

Listens on a TCP control socket for incoming connections from run_sender_eval.py.
For each connection:
  1. Receives a run_id string from the sender.
  2. Parses the category and video stem from the run_id.
  3. Launches receiver-3d.py with the correct output paths.
  4. Sends "READY" to unblock the sender.
  5. Waits for the receiver subprocess (processing and saving the video) to finish before accepting the next run.

Run ID format (set by run_sender_eval.py):
    <video_stem>_<category>_<trace_stem>
    Example: scene_01_wifi_trace_03

Output layout:
    <OUTPUT_ROOT>/receiver_logs/<category>/rgb/<video_stem>.mp4
    <OUTPUT_ROOT>/receiver_logs/<category>/depth/<video_stem>_vis.mp4
    <OUTPUT_ROOT>/receiver_logs/<category>/logs/<video_stem>.log
"""

import socket
import subprocess
import os
import sys
import time

# ──────────────────────────────────────────────────────────────────────────────
# Configuration — edit these before running
# ──────────────────────────────────────────────────────────────────────────────

HOST_IP      = "0.0.0.0"   # listen on all interfaces
CONTROL_PORT = 6000         # must match run_sender_eval.py
CODEC        = "h265"       # "h265", "h264", or "dcvcrt" (must match sender)
SERVER_IP    = "10.0.0.5"   # signaling server IPEXTRA20OFF

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
RECEIVER_SCRIPT = os.path.join(BASE_DIR, "receiver-3d.py")
OUTPUT_ROOT     = os.path.normpath(os.path.join(BASE_DIR, f"../../output/{CODEC}/receiver_logs"))

# All network category names that may appear in run IDs.
# Used to route output into the correct subdirectory.
CATEGORIES = ["cell", "wifi", "eth"]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def get_category(run_id: str) -> str:
    """
    Extract the network category from a run_id of the form
    <video_stem>_<category>_<trace_stem>.
    Returns "mixed" if no known category tag is found.
    """

    run_id_lower = run_id.lower()
    for cat in CATEGORIES:
        if f"_{cat}_" in run_id_lower:
            return cat
    return "mixed"


def get_video_stem(run_id: str) -> str:
    """
    Strip the _<category>_<trace_stem> suffix from a run_id to recover the
    original video stem.
    """
    for cat in CATEGORIES:
        tag = f"_{cat}_"
        if tag in run_id:
            return run_id.split(tag)[0]
    return run_id  # fallback; should not happen with well-formed run IDs


# ──────────────────────────────────────────────────────────────────────────────
# Service loop
# ──────────────────────────────────────────────────────────────────────────────

def run_service():
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((HOST_IP, CONTROL_PORT))
    server_sock.listen(1)

    print(f"[receiver-eval] Listening on port {CONTROL_PORT}")
    print(f"[receiver-eval] Output root: {OUTPUT_ROOT}")

    try:
        while True:
            print("\n[receiver-eval] Waiting for sender...")
            client_sock, addr = server_sock.accept()

            with client_sock:
                print(f"[receiver-eval] Connected from {addr[0]}")

                # 1. Receive the run_id sent by run_sender_eval.py
                run_id = client_sock.recv(1024).decode().strip()
                if not run_id:
                    continue

                # 2. Derive output paths from the run_id
                category   = get_category(run_id)
                video_stem = get_video_stem(run_id)

                cat_root  = os.path.join(OUTPUT_ROOT, category)
                rgb_dir   = os.path.join(cat_root, "rgb")
                depth_dir = os.path.join(cat_root, "depth")
                log_dir   = os.path.join(cat_root, "logs")
                os.makedirs(rgb_dir,   exist_ok=True)
                os.makedirs(depth_dir, exist_ok=True)
                os.makedirs(log_dir,   exist_ok=True)

                out_rgb   = os.path.join(rgb_dir,   f"{video_stem}.mp4")
                out_depth = os.path.join(depth_dir, f"{video_stem}_vis.mp4")
                log_path  = os.path.join(log_dir,   f"{video_stem}.log")

                print(f"[receiver-eval] run_id:  {run_id}")
                print(f"[receiver-eval] category: {category}  video: {video_stem}")
                print(f"[receiver-eval] rgb:   {out_rgb}")
                print(f"[receiver-eval] depth: {out_depth}")

                # 3. Launch receiver-3d.py
                cmd = [
                    sys.executable, RECEIVER_SCRIPT,
                    "--out",       out_rgb,
                    "--out_depth", out_depth,
                    "--server_ip", SERVER_IP,
                    "--codec",     CODEC,
                ]
                with open(log_path, "w") as f_log:
                    proc = subprocess.Popen(cmd, stdout=f_log, stderr=subprocess.STDOUT)
                    time.sleep(1.0)  # let the receiver connect to the signaling server

                    # 4. Signal the sender that the receiver is ready
                    client_sock.sendall(b"READY")
                    print("[receiver-eval] READY sent to sender.")

                    # 5. Wait for this run to complete before accepting the next
                    proc.wait()
                    print(f"[receiver-eval] Finished: {run_id}")

    except KeyboardInterrupt:
        print("\n[receiver-eval] Stopped.")
    finally:
        server_sock.close()


if __name__ == "__main__":
    run_service()
