import subprocess
import time
import os
import sys

def run_integration_test():
    print("=== STARTING END-TO-END RECOVERY INTEGRATION TEST ===")

    # Paths
    project_root = "/Users/kushagraaitha/Documents/ReVo"
    sig_server = os.path.join(project_root, "src/signalling_server.py")
    receiver = os.path.join(project_root, "src/receiver/receiver-3d.py")
    sender = os.path.join(project_root, "src/sender/sender-3d.py")

    rgb_video = "dummy_rgb.mp4"
    depth_video = "dummy_depth.mp4"

    # Create outputs directory
    os.makedirs(os.path.join(project_root, "output"), exist_ok=True)
    out_rgb = os.path.join(project_root, "output/out_rgb.mp4")
    out_depth = os.path.join(project_root, "output/out_depth.mp4")

    # 1. Start Signaling Server
    print("Starting Signaling Server...")
    sig_proc = subprocess.Popen(
        [sys.executable, "-u", sig_server],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    time.sleep(2) # Give it a moment to bind to port 8080

    # 2. Start Receiver (with REVO_TEST_DROP_KEYFRAME=1 and unbuffered output)
    print("Starting Receiver (with drop-keyframe=1)...")
    env_rec = os.environ.copy()
    env_rec["REVO_TEST_DROP_KEYFRAME"] = "1"
    env_rec["PYTHONUNBUFFERED"] = "1"

    receiver_log = open("output/receiver.log", "w")
    rec_proc = subprocess.Popen(
        [
            sys.executable, "-u", receiver,
            "--codec", "h265",
            "--server_ip", "127.0.0.1",
            "--out", out_rgb,
            "--out_depth", out_depth
        ],
        env=env_rec,
        stdout=receiver_log,
        stderr=receiver_log,
        text=True
    )
    time.sleep(2)

    # 3. Start Sender (with mock decoder and unbuffered output)
    print("Starting Sender (with mock decoder)...")
    env_send = os.environ.copy()
    env_send["REVO_TEST_MOCK_DEC"] = "1"
    env_send["PYTHONUNBUFFERED"] = "1"

    sender_dir = os.path.dirname(sender)
    wrapper_code = (
        "import sys, types, runpy\n"
        f"sys.path.insert(0, '{sender_dir}')\n"
        "torchcodec = types.ModuleType('torchcodec')\n"
        "torchcodec.decoders = types.ModuleType('torchcodec.decoders')\n"
        "class MockVideoDecoder:\n"
        "    def __init__(self, file_path, device='cpu'): self.num_frames = 60\n"
        "    def __len__(self): return self.num_frames\n"
        "    def __getitem__(self, idx):\n"
        "        import torch\n"
        "        return torch.zeros((3, 512, 512), dtype=torch.uint8)\n"
        "torchcodec.decoders.VideoDecoder = MockVideoDecoder\n"
        "sys.modules['torchcodec'] = torchcodec\n"
        "sys.modules['torchcodec.decoders'] = torchcodec.decoders\n"
        f"runpy.run_path('{sender}', run_name='__main__')\n"
    )

    sender_log = open("output/sender.log", "w")
    send_proc = subprocess.Popen(
        [
            sys.executable, "-u", "-c", wrapper_code,
            "--codec", "h265",
            "--server_ip", "127.0.0.1",
            "--file", rgb_video,
            "--depth_file", depth_video
        ],
        env=env_send,
        stdout=sender_log,
        stderr=sender_log,
        text=True
    )

    # Wait for sender to stream all frames (the mock has 60 frames)
    print("Streaming in progress...")
    try:
        send_proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        print("Sender timed out! Terminating processes...")
        send_proc.terminate()

    time.sleep(2)
    rec_proc.terminate()
    sig_proc.terminate()

    receiver_log.close()
    sender_log.close()

    # Read logs and verify recovery
    with open("output/receiver.log", "r") as f:
        rec_content = f.read()
    with open("output/sender.log", "r") as f:
        send_content = f.read()

    print("\n=== RECEIVER LOG EXCERPTS ===")
    for line in rec_content.splitlines():
        if "[Recovery]" in line or "[Feedback]" in line or "frozen" in line or "Decode" in line or "[DEBUG]" in line:
            print(f"Receiver: {line}")

    print("\n=== SENDER LOG EXCERPTS ===")
    for line in send_content.splitlines():
        if "[Feedback]" in line or "Forcing" in line:
            print(f"Sender: {line}")

    # Verify expectations
    print("\nVerifying integration metrics...")

    # A. Did the receiver request a keyframe for frame 30?
    req_sent = "[Recovery] Sent KEYFRAME_REQUEST for lost keyframe 30" in rec_content
    # B. Did the sender receive the request?
    req_received = "Received KEYFRAME_REQUEST for frame 30" in send_content
    # C. Did the sender force the next frame to be a keyframe?
    forced_sent = "Forcing keyframe for frame" in send_content
    # D. Did the receiver successfully decode the forced keyframe at Frame 33?
    recovered = "[Recovery] Successfully decoded keyframe 33." in rec_content

    print(f"1. KEYFRAME_REQUEST sent: {req_sent}")
    print(f"2. KEYFRAME_REQUEST received: {req_received}")
    print(f"3. Keyframe forced: {forced_sent}")
    print(f"4. Receiver recovered at frame 33: {recovered}")

    assert req_sent, "Test Failed: Receiver did not request keyframe!"
    assert req_received, "Test Failed: Sender did not receive request!"
    assert forced_sent, "Test Failed: Sender did not force keyframe!"
    assert recovered, "Test Failed: Receiver did not recover at frame 33!"

    print("\n=== END-TO-END INTEGRATION TEST PASSED SUCCESSFULLY! ===")

if __name__ == "__main__":
    run_integration_test()
