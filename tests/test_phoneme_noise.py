"""Unit tests for the phoneme-noise augmenter."""

from __future__ import annotations

import os
import sys
import unittest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from Data.phoneme_noise import (
    DEFAULT_VISUAL_CONFUSION_MAP,
    PhonemeNoiseAugmenter,
    build_augmenter_from_config,
)


SAMPLE = ["DH", "AH", "K", "AE", "T"]


class PhonemeNoiseAugmenterTests(unittest.TestCase):
    def test_identity_when_all_probs_are_zero(self):
        aug = PhonemeNoiseAugmenter(substitute_prob=0.0, insert_prob=0.0, delete_prob=0.0)
        self.assertEqual(aug(SAMPLE), SAMPLE)

    def test_deterministic_under_seed(self):
        a = PhonemeNoiseAugmenter(seed=7, substitute_prob=0.5, insert_prob=0.2, delete_prob=0.1)
        b = PhonemeNoiseAugmenter(seed=7, substitute_prob=0.5, insert_prob=0.2, delete_prob=0.1)
        # Same seed → same output over many sequences.
        for seq in (SAMPLE, ["B", "IY"], ["IH", "NG", "K"]):
            self.assertEqual(a(seq), b(seq))

    def test_substitution_uses_confusion_map(self):
        aug = PhonemeNoiseAugmenter(
            substitute_prob=1.0, insert_prob=0.0, delete_prob=0.0,
            confusion_map={"B": [("P", 1.0)]},
            fallback_vocab=["B", "P"],
            seed=0,
        )
        self.assertEqual(aug(["B"]), ["P"])

    def test_substitution_weights_are_normalised(self):
        aug = PhonemeNoiseAugmenter(
            substitute_prob=1.0, insert_prob=0.0, delete_prob=0.0,
            confusion_map={"B": [("P", 3.0), ("M", 1.0)]},
            fallback_vocab=["B", "P", "M"],
            seed=0,
        )
        n = 500
        counts = {"P": 0, "M": 0}
        for _ in range(n):
            out = aug(["B"])
            self.assertEqual(len(out), 1)
            counts[out[0]] += 1
        # P should be ~3x more likely than M.
        ratio = counts["P"] / max(1, counts["M"])
        self.assertGreater(ratio, 1.8)
        self.assertLess(ratio, 5.0)

    def test_insertion_can_grow_sequence(self):
        aug = PhonemeNoiseAugmenter(
            substitute_prob=0.0, insert_prob=1.0, delete_prob=0.0,
            seed=1,
        )
        out = aug(["AH"])
        self.assertEqual(len(out), 2)
        self.assertEqual(out[-1], "AH")

    def test_deletion_can_shrink_sequence(self):
        aug = PhonemeNoiseAugmenter(
            substitute_prob=0.0, insert_prob=0.0, delete_prob=1.0,
            seed=1,
        )
        self.assertEqual(aug(SAMPLE), [])

    def test_build_augmenter_from_config_disabled(self):
        self.assertIsNone(build_augmenter_from_config(None))
        self.assertIsNone(build_augmenter_from_config({"enabled": False}))

    def test_build_augmenter_from_config_enabled(self):
        aug = build_augmenter_from_config({"enabled": True, "seed": 0})
        self.assertIsNotNone(aug)
        # Runs cleanly on the default vocab.
        out = aug(SAMPLE)
        self.assertIsInstance(out, list)

    def test_default_map_has_expected_visemic_pairs(self):
        # Some sanity checks on the default map.
        b_map = dict(DEFAULT_VISUAL_CONFUSION_MAP["B"])
        self.assertIn("P", b_map)
        p_map = dict(DEFAULT_VISUAL_CONFUSION_MAP["P"])
        self.assertIn("B", p_map)
        f_map = dict(DEFAULT_VISUAL_CONFUSION_MAP["F"])
        self.assertIn("V", f_map)


if __name__ == "__main__":
    unittest.main()
