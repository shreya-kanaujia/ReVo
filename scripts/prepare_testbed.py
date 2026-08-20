#!/usr/bin/env python3
import os
import subprocess

def main():
    print("=== PREPARING REVO LOCAL TESTBED ===")

    # 1. Ensure directories exist
    os.makedirs("output", exist_ok=True)
    os.makedirs("src/sender/traces", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    # 2. Generate trace file
    trace_path = "src/sender/traces/constant_loss_10pct_360s.log"
    print(f"Generating trace file: {trace_path}")
    with open(trace_path, "w") as f:
        f.write("0.000 1000.0 0 0.10\n")
        f.write("360.000 1000.0 0 0.10\n")

    # 3. Build Docker image
    print("Building Docker image revo-local-testbed...")
    subprocess.run([
        "docker", "build", "-f", "Dockerfile.local-testbed", "-t", "revo-local-testbed", "."
    ], check=True)

    # 4. Slice video files using FFmpeg in the container
    print("Slicing RGB video...")
    subprocess.run([
        "docker", "run", "--rm", "-v", f"{os.getcwd()}:/app", "revo-local-testbed",
        "ffmpeg", "-y", "-i", "data/gt_rgb/1lSejjfNHpw_0075_S0_E728_L671_T47_R1471_B847.mp4",
        "-t", "5", "-c", "copy", "data/short_rgb.mp4"
    ], check=True)

    print("Slicing depth video...")
    subprocess.run([
        "docker", "run", "--rm", "-v", f"{os.getcwd()}:/app", "revo-local-testbed",
        "ffmpeg", "-y", "-i", "data/gt_depth/1lSejjfNHpw_0075_S0_E728_L671_T47_R1471_B847_vis.mp4",
        "-t", "5", "-c", "copy", "data/short_depth.mp4"
    ], check=True)

    print("=== TESTBED PREPARATION COMPLETE ===")

if __name__ == "__main__":
    main()
