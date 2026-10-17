#!/usr/bin/env python3
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


summary = load("scale_summary", HERE / "summarize.py")
evaluator = load("intervals", ROOT / "tools/evaluate-learning.py")
amendment = json.loads((HERE / "confirmation-amendment.json").read_text())


class ConfirmationTests(unittest.TestCase):
    def test_discovery_cannot_supply_positive_evidence(self):
        result = summary.confirmation_analysis([(0, [1.0]*64)], amendment, evaluator)
        self.assertFalse(result["positive"])
        self.assertEqual(result["analysis_games"], 0)
        self.assertEqual(result["pair_scores"], [])

    def test_excluded_outcomes_do_not_change_the_inference(self):
        low = summary.confirmation_analysis([(0, [0.0]*64), (1, [0.0]+[.5]*63)], amendment, evaluator)
        high = summary.confirmation_analysis([(0, [1.0]*64), (1, [1.0]+[.5]*63)], amendment, evaluator)
        self.assertEqual(low, high)
        self.assertEqual(low["analysis_games"], 126)
        self.assertEqual(low["score"], .5)
        self.assertFalse(low["positive"])

    def test_first_positive_prefix_survives_unneeded_later_results(self):
        result = summary.confirmation_analysis([(1, [.5]+[1.0]*63), (2, [.5]+[0.0]*63)], amendment, evaluator)
        self.assertTrue(result["positive"])
        self.assertEqual(result["first_positive_checkpoint"]["block"], 1)
        self.assertEqual(result["available_confirmation_games"], 252)
        self.assertEqual(result["analysis_games"], 126)
        self.assertEqual(result["score"], 1.0)
        self.assertLess(result["confirmation_checkpoints"][-1]["betting_interval"][0], .5)

    def test_all_retained_games_have_distinct_competitor_seeds(self):
        protocol = json.loads((HERE / "protocol.json").read_text())
        mask = (1 << 64)-1
        for block in protocol["blocks"]:
            collisions = []
            for i in range(protocol["games_per_block"]):
                a = block["seed"] ^ ((i*0xA24BAED4963EE407) & mask)
                b = block["seed"] ^ ((i*0x9FB21C651E98DF25) & mask)
                if a == b:
                    collisions.append(i)
                if block["index"] in amendment["confirmation_block_indices"] and i//2 not in amendment["exclude_pair_indices_per_confirmation_block"]:
                    self.assertNotEqual(a, b)
            self.assertEqual(collisions, [0])

    def test_original_registration_and_discovery_are_preserved(self):
        for name, expected in [("protocol.json", amendment["original_protocol_sha256"]),
                               ("results/block-00-raw.jsonl", amendment["discovery_raw_sha256"])]:
            self.assertEqual(hashlib.sha256((HERE/name).read_bytes()).hexdigest(), expected)
        self.assertEqual(amendment["max_total_executed_games"], 2048)
        self.assertEqual(amendment["max_confirmation_games"], 1890)
        self.assertEqual(amendment["completed_blocks_when_registered"], [0])


if __name__ == "__main__":
    unittest.main()
