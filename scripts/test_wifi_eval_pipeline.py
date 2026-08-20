#!/usr/bin/env python3
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wifi_eval_runner as runner
from generate_frame_masks import parse_log_file


class WifiEvaluationPipelineTests(unittest.TestCase):
    def test_smoke_has_frame_cap_and_full_run_does_not(self):
        smoke = runner.runtime_config("baseline", True, 150, 150)
        full = runner.runtime_config("baseline", False, 150, 8640)
        self.assertEqual(smoke["REVO_MAX_FRAMES"], "150")
        self.assertNotIn("REVO_MAX_FRAMES", full)
        self.assertEqual(full["REVO_EVAL_FULL_RUN"], "1")
        self.assertEqual(smoke["REVO_EVAL_FULL_RUN"], "0")

    def test_runtime_identity_hashes_are_frozen(self):
        config = runner.runtime_config("baseline", False, 150, 8640)
        for key, path in runner.RUNTIME_IDENTITY_FILES.items():
            self.assertEqual(config[key], runner.sha256_file(path))

    def test_runtime_preflight_rejects_mismatch(self):
        failed = mock.Mock(returncode=1, stdout="", stderr="identity mismatch")
        with mock.patch.object(runner.subprocess, "run", return_value=failed):
            with self.assertRaisesRegex(RuntimeError, "runtime preflight failed"):
                runner.verify_runtime_container({})

    def test_full_run_removes_inherited_smoke_cap(self):
        mapping = runner.parse_trace_map()[0]
        env = runner.build_run_environment(
            "baseline", False, 150, 8640, mapping,
            base_environment={"REVO_MAX_FRAMES": "150"},
        )
        self.assertNotIn("REVO_MAX_FRAMES", env)
        self.assertEqual(env["RGB_SOURCE"], mapping["rgb_source"])
        self.assertEqual(env["DEPTH_SOURCE"], mapping["depth_source"])
        self.assertEqual(env["TRACE_PATH"], mapping["trace_path"])

    def test_timeout_policy_differs(self):
        self.assertEqual(runner.timeout_for_run(True, 345.0), runner.SMOKE_TIMEOUT_S)
        self.assertGreater(runner.timeout_for_run(False, 345.0), 345.0)
        self.assertGreater(runner.timeout_for_run(False, 345.0), 180.0)

    def test_source_window_excludes_post_source_fid(self):
        with tempfile.NamedTemporaryFile("w", suffix=".log") as log:
            for fid in (0, 30, 60, 90, 120):
                log.write(f"[FID: {fid}] Successfully decoded keyframe {fid}.\n")
            log.write("[FID: 149] frame ok\n")
            log.write("[FID: 150] I-frame not fully ready. Dropping.\n")
            log.flush()
            _, mask = parse_log_file(log.name, max_frames=150)
        self.assertEqual(len(mask), 150)
        self.assertEqual(int(mask.sum()), 0)

    def test_resume_rejects_incomplete_and_accepts_validated(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            artifacts = {}
            for name in ("rgb", "depth", "receiver_log", "sender_log", "tc", "mask"):
                path = tmp / name
                path.write_bytes(b"x")
                artifacts[name] = str(path)
            marker = tmp / "status.json"
            marker.write_text(json.dumps({
                "status": "INCOMPLETE", "artifacts": artifacts,
                "validation": {"expected_output_frames": 150},
            }))
            self.assertFalse(runner.success_marker_valid(marker))
            marker.write_text(json.dumps({
                "status": "VALIDATED_COMPLETE", "artifacts": artifacts,
                "validation": {"expected_output_frames": 150},
            }))
            with mock.patch.object(runner, "video_info", return_value={"frames": 150, "fps": 25.0}), \
                 mock.patch.object(runner, "validate_tc_evidence", return_value={"bytes": 1}):
                self.assertTrue(runner.success_marker_valid(marker))
            Path(artifacts["depth"]).unlink()
            self.assertFalse(runner.success_marker_valid(marker))

    def test_manifest_contains_both_method_configs(self):
        mapping = runner.parse_trace_map()[0]
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(runner, "video_info", return_value={"frames": 150, "fps": 25.0}), \
             mock.patch.object(runner, "sha256_file", return_value="hash"), \
             mock.patch.object(runner, "get_git_commit", return_value="commit"):
            root = Path(tmp)
            runner.ensure_manifest(root, [mapping])
            baseline = json.loads((root / "manifest/baseline_config.json").read_text())
            abr = json.loads((root / "manifest/abr_config.json").read_text())
            manifest = json.loads((root / "manifest/experiment_manifest.json").read_text())
            self.assertEqual(baseline["configuration"]["SALSIFY_MODE"], "0")
            self.assertEqual(abr["configuration"]["SALSIFY_MODE"], "1")
            self.assertEqual(baseline["configuration"]["SALSIFY_DUP_DELAY_MS"], "33")
            self.assertEqual(manifest["wifi_mapping_count"], 1)

    def test_tc_evidence_accepts_zero_drop_smoke(self):
        with tempfile.NamedTemporaryFile("w", suffix=".log") as evidence:
            evidence.write("capture_exit_status=0\nqdisc htb 1: root\n")
            evidence.write("qdisc netem 10: parent 1:10 limit 1000 delay 40ms loss 0%\n")
            evidence.write(" Sent 1000 bytes 10 pkt (dropped 0, overlimits 0 requeues 0)\n")
            evidence.flush()
            parsed = runner.validate_tc_evidence(evidence.name)
        self.assertEqual(parsed["drops"], 0)
        self.assertEqual(parsed["packets"], 10)

    def test_shared_configuration_is_explicit(self):
        for method in ("baseline", "abr"):
            config = runner.method_config(method)
            for key in ("REVO_GOP", "SALSIFY_FEC_ADAPT", "SALSIFY_FEC_MIN",
                        "SALSIFY_FEC_MAX", "SALSIFY_FEC_MAXPK", "REVO_KF_RETRY_S",
                        "REVO_PLAYOUT_MS", "REVO_FB_INTERVAL", "REVO_FB_WINDOW",
                        "SALSIFY_DUP_DELAY_MS"):
                self.assertIn(key, config)


if __name__ == "__main__":
    unittest.main()
