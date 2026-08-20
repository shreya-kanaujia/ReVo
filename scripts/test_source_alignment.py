#!/usr/bin/env python3
import os
import sys
import tempfile
import numpy as np
import torch
import av

# Add source directories to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src/lossrec/rgb")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src/lossrec/depth")))

from dataloader_finetune_inference import LossRecoveryData as RgbDataloader
from dataloader_finetune_inference_depth import LossRecoveryData as DepthDataloader

def write_dummy_video(path, num_frames, fill_values):
    container = av.open(path, mode="w")
    stream = container.add_stream("libx265", rate=25)
    stream.width = 224
    stream.height = 224
    stream.pix_fmt = "yuv420p"
    stream.options = {"preset": "ultrafast", "crf": "23"}

    for i in range(num_frames):
        val = fill_values[i]
        arr = np.full((224, 224, 3), val, dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)
    container.close()

def run_alignment_test():
    print("=== STARTING DETERMINISTIC SOURCE-FRAME ALIGNMENT TESTS ===")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create directories
        clean_dir = os.path.join(tmpdir, "gt")
        corrupted_dir = os.path.join(tmpdir, "corrupted")
        mask_dir = os.path.join(tmpdir, "masks")
        log_dir = os.path.join(tmpdir, "logs")

        for d in [clean_dir, corrupted_dir, mask_dir, log_dir]:
            os.makedirs(d, exist_ok=True)

        num_clean = 60
        num_corr = 30
        fid0 = 30
        corrupted_clean_fid = 35
        corrupted_corr_fid = corrupted_clean_fid - fid0 # 5
        
        clean_vals = list(range(num_clean))
        corr_vals = [fid0 + j for j in range(num_corr)]
        corr_vals[corrupted_corr_fid] = 99 # Inject unique corrupted value

        # ==========================================
        # 1. SETUP RGB TEST ASSETS
        # ==========================================
        clean_rgb_path = os.path.join(clean_dir, "test_rgb_video.mp4")
        corr_rgb_path = os.path.join(corrupted_dir, "test_rgb_video.mp4")
        write_dummy_video(clean_rgb_path, num_clean, clean_vals)
        write_dummy_video(corr_rgb_path, num_corr, corr_vals)

        mask_rgb = np.zeros(num_clean, dtype=int)
        mask_rgb[corrupted_clean_fid] = 1
        np.save(os.path.join(mask_dir, "test_rgb_video_frame_mask.npy"), mask_rgb)

        with open(os.path.join(log_dir, "test_rgb_video.log"), "w") as f:
            f.write(f"[Recovery] Successfully decoded keyframe {fid0}. Cleared pending request state.\n")

        # ==========================================
        # 2. SETUP DEPTH TEST ASSETS
        # ==========================================
        clean_depth_path = os.path.join(clean_dir, "test_depth_video_vis.mp4")
        corr_depth_path = os.path.join(corrupted_dir, "test_depth_video_vis.mp4")
        write_dummy_video(clean_depth_path, num_clean, clean_vals)
        write_dummy_video(corr_depth_path, num_corr, corr_vals)

        mask_depth = np.zeros(num_clean, dtype=int)
        mask_depth[corrupted_clean_fid] = 1
        np.save(os.path.join(mask_dir, "test_depth_video_frame_mask.npy"), mask_depth)

        with open(os.path.join(log_dir, "test_depth_video.log"), "w") as f:
            f.write(f"[Recovery] Successfully decoded keyframe {fid0}. Cleared pending request state.\n")

        print("Created synthetic video assets and logs.")

        # ==========================================
        # 3. TEST RGB DATALOADER ALIGNMENT
        # ==========================================
        print("Testing RGB Dataloader alignment...")
        rgb_dataset = RgbDataloader(
            clean_dir=clean_dir,
            corrupted_dir=corrupted_dir,
            mask_dir=mask_dir,
            num_frames=6,
            input_size=224,
            log_dir=log_dir
        )

        assert len(rgb_dataset) > 0, "RGB Dataset should have at least 1 clip"
        
        rgb_clip = None
        for i in range(len(rgb_dataset)):
            (corrupted_clip, clip_mask), clean_clip, meta, frame_indices = rgb_dataset[i]
            if meta["video_name"] == "test_rgb_video":
                rgb_clip = (corrupted_clip, clean_clip, meta)
                break
        
        assert rgb_clip is not None, "Could not find test_rgb_video clip"
        corrupted_clip, clean_clip, meta = rgb_clip

        # Verify target frame metadata
        assert meta["start_frame"] == corrupted_corr_fid, f"Expected target start frame {corrupted_corr_fid}, got {meta['start_frame']}"

        # Decode directly to compare values
        from torchcodec.decoders import VideoDecoder
        clean_dec = VideoDecoder(clean_rgb_path, device="cpu")
        corr_dec = VideoDecoder(corr_rgb_path, device="cpu")

        # Transform reference frames
        expected_clean_tensor = rgb_dataset.transform(clean_dec[corrupted_clean_fid])
        expected_corr_tensor = rgb_dataset.transform(corr_dec[corrupted_corr_fid])

        clean_target_val = float(clean_clip[0, -1, 0, 0].item())
        expected_clean_val = float(expected_clean_tensor[0, 0, 0].item())
        assert abs(clean_target_val - expected_clean_val) < 1e-6, f"Clean target frame value mismatch: expected {expected_clean_val}, got {clean_target_val}"

        corr_target_val = float(corrupted_clip[0, -1, 0, 0].item())
        expected_corr_val = float(expected_corr_tensor[0, 0, 0].item())
        assert abs(corr_target_val - expected_corr_val) < 1e-6, f"Corrupted target frame value mismatch: expected {expected_corr_val}, got {corr_target_val}"

        print("  → RGB alignment PASS!")

        # ==========================================
        # 4. TEST DEPTH DATALOADER ALIGNMENT
        # ==========================================
        print("Testing Depth Dataloader alignment...")
        depth_dataset = DepthDataloader(
            clean_dir=clean_dir,
            corrupted_dir=corrupted_dir,
            mask_dir=mask_dir,
            num_frames=6,
            input_size=224,
            log_dir=log_dir
        )

        assert len(depth_dataset) > 0, "Depth Dataset should have at least 1 clip"
        
        depth_clip = None
        for i in range(len(depth_dataset)):
            (corrupted_clip, clip_mask), clean_clip, meta, frame_indices = depth_dataset[i]
            if meta["video_name"] == "test_depth_video_vis":
                depth_clip = (corrupted_clip, clean_clip, meta)
                break

        assert depth_clip is not None, "Could not find test_depth_video_vis clip"
        corrupted_clip, clean_clip, meta = depth_clip

        # Verify target frame metadata
        assert meta["start_frame"] == corrupted_corr_fid, f"Expected target start frame {corrupted_corr_fid}, got {meta['start_frame']}"

        # Decode directly to compare values
        clean_dec_depth = VideoDecoder(clean_depth_path, device="cpu")
        corr_dec_depth = VideoDecoder(corr_depth_path, device="cpu")

        # Transform reference frames
        expected_clean_tensor_depth = depth_dataset.transform(clean_dec_depth[corrupted_clean_fid])
        expected_corr_tensor_depth = depth_dataset.transform(corr_dec_depth[corrupted_corr_fid])

        clean_target_val_depth = float(clean_clip[0, -1, 0, 0].item())
        expected_clean_val_depth = float(expected_clean_tensor_depth[0, 0, 0].item())
        assert abs(clean_target_val_depth - expected_clean_val_depth) < 1e-6, f"Depth clean target frame value mismatch: expected {expected_clean_val_depth}, got {clean_target_val_depth}"

        corr_target_val_depth = float(corrupted_clip[0, -1, 0, 0].item())
        expected_corr_val_depth = float(expected_corr_tensor_depth[0, 0, 0].item())
        assert abs(corr_target_val_depth - expected_corr_val_depth) < 1e-6, f"Depth corrupted target frame value mismatch: expected {expected_corr_val_depth}, got {corr_target_val_depth}"

        print("  → Depth alignment PASS!")

    print("==========================================================")
    print("RESULT: ALL SOURCE-FRAME ALIGNMENT TESTS PASSED!")
    sys.exit(0)

if __name__ == "__main__":
    run_alignment_test()
