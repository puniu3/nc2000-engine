#!/usr/bin/env python3
"""Execution-control tests, isolated from live experiment artifacts/processes."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

spec = importlib.util.spec_from_file_location("advance", Path(__file__).with_name("advance.py"))
advance = importlib.util.module_from_spec(spec)
spec.loader.exec_module(advance)


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.here = self.root / "findings/headroom-scale-v1"
        (self.here / "results").mkdir(parents=True)
        binary = self.root / "target/release/examples/arena"
        binary.parent.mkdir(parents=True)
        binary.write_bytes(b"frozen executable")
        (self.here / "results/block-00.json").write_text(json.dumps({
            "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest()}))
        (self.here / "protocol.json").write_text(json.dumps({"blocks": list(range(16))}))
        self.previous = self.here / "previous.stdout.jsonl"
        self.previous.write_text(json.dumps({"type": "segment_complete", "last_index": 3}) + "\n")
        for name, value in [("HERE", self.here), ("ROOT", self.root)]:
            patch = mock.patch.object(advance, name, value)
            patch.start()
            self.addCleanup(patch.stop)
        self.argv = mock.patch.object(advance.sys, "argv", ["advance.py", "--wait-pid", "123",
                                      "--previous-stdout", str(self.previous)])
        self.argv.start()
        self.addCleanup(self.argv.stop)
        patch = mock.patch.object(advance.subprocess, "run", return_value=SimpleNamespace(returncode=1))
        patch.start()
        self.addCleanup(patch.stop)

    def summary(self, blocks, positive=False):
        return {"completed_blocks": blocks, "games": blocks*128, "positive": positive,
                "score": .6 if positive else .5, "betting_interval": [.51, .69] if positive else [.4, .6]}

    def test_positive_checkpoint_and_cap_do_not_launch_more_games(self):
        for blocks, positive, expected in [(4, True, "positive_checkpoint_verified"),
                                           (16, False, "registered_sample_cap_reached")]:
            self.previous.write_text(json.dumps({"type": "segment_complete", "last_index": blocks-1}))
            with mock.patch.object(advance, "audit", return_value=self.summary(blocks, positive)), \
                    mock.patch.object(advance.subprocess, "Popen") as launch, \
                    mock.patch.object(advance, "state") as status:
                advance.main()
                launch.assert_not_called()
                self.assertEqual(status.call_args.args[0], expected)

    def test_failed_or_partial_segment_is_not_restarted(self):
        self.previous.write_text(json.dumps({"type": "checkpoint", "index": 2}))
        with mock.patch.object(advance.subprocess, "Popen") as launch:
            with self.assertRaisesRegex(AssertionError, "did not finish cleanly"):
                advance.main()
            launch.assert_not_called()

    def test_changed_executable_is_rejected_before_new_games(self):
        (self.root / "target/release/examples/arena").write_bytes(b"changed")
        with mock.patch.object(advance, "audit", return_value=self.summary(4)), \
                mock.patch.object(advance.subprocess, "Popen") as launch:
            with self.assertRaisesRegex(AssertionError, "executable changed"):
                advance.main()
            launch.assert_not_called()

    def test_existing_output_is_preserved(self):
        evidence = self.here / "execution-from-04.stdout.jsonl"
        evidence.write_text("existing evidence")
        with mock.patch.object(advance, "audit", return_value=self.summary(4)), \
                mock.patch.object(advance.subprocess, "Popen") as launch:
            with self.assertRaises(FileExistsError):
                advance.main()
            launch.assert_not_called()
        self.assertEqual(evidence.read_text(), "existing evidence")

    def test_only_next_registered_segment_runs_then_positive_stops(self):
        def launch(command, **kwargs):
            self.assertEqual(command[-6:], ["--start", "4", "--count", "4", "--threads", "10"])
            kwargs["stdout"].write(json.dumps({"type": "segment_complete", "last_index": 7}) + "\n")
            return SimpleNamespace(pid=456, wait=lambda: 0)
        with mock.patch.object(advance, "audit", side_effect=[self.summary(4), self.summary(8, True)]), \
                mock.patch.object(advance.subprocess, "Popen", side_effect=launch) as launched, \
                mock.patch.object(advance, "state") as status:
            advance.main()
            self.assertEqual(launched.call_count, 1)
            self.assertEqual(status.call_args.args[0], "positive_checkpoint_verified")


if __name__ == "__main__":
    unittest.main()
