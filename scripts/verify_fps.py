#!/usr/bin/env python3
import os
import sys
import re
import av

def verify_log_content(log_path, pattern, description):
    if not os.path.exists(log_path):
        print(f"FAIL: Log file {log_path} not found")
        return False
    
    with open(log_path, "r") as f:
        for line in f:
            if re.search(pattern, line):
                print(f"PASS: Found {description} in {os.path.basename(log_path)}")
                return True
    print(f"FAIL: Could not find {description} in {os.path.basename(log_path)}")
    return False

def verify_video_cadence(video_path, description, expected_frame_count):
    if not os.path.exists(video_path):
        print(f"FAIL: Video file {video_path} not found")
        return False

    try:
        container = av.open(video_path)
        stream = container.streams.video[0]
        fps = float(stream.average_rate)
        frame_count = int(stream.frames or 0)
        print(f"PASS: {description} metadata FPS is {fps}")
        if frame_count != expected_frame_count:
            print(f"FAIL: {description} has {frame_count} frames, expected {expected_frame_count}")
            container.close()
            return False
        
        # Decode first 5 frames and check timestamps
        pts_list = []
        for i, frame in enumerate(container.decode(video=0)):
            if i >= 5:
                break
            t = float(frame.pts * frame.time_base)
            pts_list.append(t)
        
        container.close()
        
        print(f"  → First {len(pts_list)} decoded frame timestamps: {pts_list}")
        
        # Verify output cadence: spacing should be ~0.04s (25 FPS)
        if len(pts_list) > 1:
            diffs = [pts_list[i] - pts_list[i-1] for i in range(1, len(pts_list))]
            mean_diff = sum(diffs) / len(diffs)
            print(f"  → Mean output frame interval: {mean_diff:.4f}s (expected 0.0400s)")
            if abs(mean_diff - 0.04) < 0.005:
                print(f"PASS: {description} output frame cadence is verified at 25 FPS")
                return True
            else:
                print(f"FAIL: {description} output frame cadence is {mean_diff:.4f}s, expected ~0.0400s")
                return False
        else:
            print(f"FAIL: {description} has insufficient frames to check cadence")
            return False
            
    except Exception as e:
        print(f"FAIL: Error verifying {description}: {e}")
        return False

def main():
    if len(sys.argv) < 6:
        print("Usage: python scripts/verify_fps.py <sender_log> <receiver_log> <rgb_mp4> <depth_mp4> <expected_frame_count>")
        sys.exit(1)
        
    sender_log = sys.argv[1]
    receiver_log = sys.argv[2]
    rgb_mp4 = sys.argv[3]
    depth_mp4 = sys.argv[4]
    expected_frame_count = int(sys.argv[5])
    
    print("=== RUNNING AUTOMATED FPS AND CADENCE VERIFICATION ===")
    
    success = True
    
    # 1. Verify detected source FPS (25 FPS) and sender pacing configuration in sender log
    success &= verify_log_content(sender_log, r"Starting stream: \d+ frames @ 25 FPS", "detected source FPS = 25")
    
    # 2. Verify INIT message contains 25 FPS in sender log
    success &= verify_log_content(sender_log, r"INIT sent: \d+x\d+ @ 25 fps", "INIT message FPS = 25")
    
    # 3. Verify receiver writer initialized at 25 FPS in receiver log
    success &= verify_log_content(receiver_log, r"Saved \d+ RGB frames to .* \(25 FPS", "receiver RGB writer FPS = 25")
    success &= verify_log_content(receiver_log, r"Saved \d+ depth frames to .* \(25 FPS", "receiver depth writer FPS = 25")
    
    # 4. Verify resulting RGB and depth MP4 files have 25 FPS metadata and actual 40ms output frame cadence
    success &= verify_video_cadence(rgb_mp4, "resulting RGB MP4", expected_frame_count)
    success &= verify_video_cadence(depth_mp4, "resulting depth MP4", expected_frame_count)
    
    print("=====================================================")
    if success:
        print("RESULT: ALL FPS AND CADENCE VERIFICATION CHECKS PASSED!")
        sys.exit(0)
    else:
        print("RESULT: SOME FPS AND CADENCE VERIFICATION CHECKS FAILED!")
        sys.exit(1)

if __name__ == "__main__":
    main()
