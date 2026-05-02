"""Unit tests for the top-k CTC decoder and prompt formatters."""

from __future__ import annotations

import math
import os
import sys
import unittest

import torch
import torch.nn.functional as F

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from Models.ctc_decode import (
    CTCHypothesis,
    decode_and_format,
    format_one_best_prompt,
    format_topk_prompt,
    greedy_decode,
    topk_beam_decode,
)


def _make_logits(vocab_size: int, sequence):
    """Build log-probs that strongly prefer the given id sequence.

    Each timestep gets one dominant class and zero-ish mass elsewhere.
    """
    T = len(sequence)
    logits = torch.full((T, vocab_size), -5.0)
    for t, cls in enumerate(sequence):
        logits[t, cls] = 5.0
    # log_softmax so we get proper log-probs.
    return F.log_softmax(logits, dim=-1)


class GreedyDecodeTests(unittest.TestCase):
    def test_blanks_and_repeats_collapse(self):
        vocab = {"<pad>": 0, "A": 1, "B": 2, "C": 3}
        id_to = {v: k for k, v in vocab.items()}
        log_probs = _make_logits(len(vocab), [1, 1, 0, 2, 2, 3, 0, 3])
        hyps = greedy_decode(log_probs, id_to_phoneme=id_to, blank_id=0)
        self.assertEqual(len(hyps), 1)
        self.assertEqual(hyps[0].phonemes, ["A", "B", "C", "C"])

    def test_empty_sequence(self):
        vocab = {"<pad>": 0, "A": 1}
        id_to = {v: k for k, v in vocab.items()}
        log_probs = _make_logits(len(vocab), [0, 0, 0])
        hyps = greedy_decode(log_probs, id_to_phoneme=id_to, blank_id=0)
        self.assertEqual(hyps[0].phonemes, [])


class TopKBeamDecodeTests(unittest.TestCase):
    def test_topk_returns_sorted_hypotheses(self):
        vocab = {"<pad>": 0, "A": 1, "B": 2}
        id_to = {v: k for k, v in vocab.items()}
        # A slightly ambiguous sequence: mostly A, but with some B mass.
        T, V = 6, len(vocab)
        base = torch.full((T, V), -3.0)
        # Strong preference for A everywhere except t=2 where A and B tie.
        for t in range(T):
            base[t, 1] = 1.0
        base[2, 2] = 1.0
        log_probs = F.log_softmax(base, dim=-1)
        hyps = topk_beam_decode(
            log_probs, id_to_phoneme=id_to, blank_id=0,
            beam_width=8, top_k=3, prune_log_prob=-math.inf,
        )[0]
        self.assertGreaterEqual(len(hyps), 1)
        # Scores must be sorted.
        for a, b in zip(hyps, hyps[1:]):
            self.assertGreaterEqual(a.score, b.score)
        # Top-1 should be reasonable (either "A" or "A B A" depending on ties).
        self.assertIn("A", hyps[0].phonemes)

    def test_topk_agrees_with_greedy_at_top1(self):
        vocab = {"<pad>": 0, "A": 1, "B": 2, "C": 3}
        id_to = {v: k for k, v in vocab.items()}
        log_probs = _make_logits(len(vocab), [1, 1, 0, 2, 2, 3])
        greedy = greedy_decode(log_probs, id_to_phoneme=id_to, blank_id=0)[0]
        topk = topk_beam_decode(
            log_probs, id_to_phoneme=id_to, blank_id=0,
            beam_width=4, top_k=3, prune_log_prob=-math.inf,
        )[0]
        self.assertEqual(greedy.phonemes, topk[0].phonemes)

    def test_batch_dimension(self):
        vocab = {"<pad>": 0, "A": 1, "B": 2}
        id_to = {v: k for k, v in vocab.items()}
        log_probs_a = _make_logits(len(vocab), [1, 0, 2])
        log_probs_b = _make_logits(len(vocab), [2, 2, 1])
        batch = torch.stack([log_probs_a, log_probs_b], dim=0)
        hyps = topk_beam_decode(
            batch, id_to_phoneme=id_to, blank_id=0,
            beam_width=4, top_k=2, prune_log_prob=-math.inf,
        )
        self.assertEqual(len(hyps), 2)
        self.assertEqual(hyps[0][0].phonemes, ["A", "B"])
        self.assertEqual(hyps[1][0].phonemes, ["B", "A"])

    def test_invalid_topk_raises(self):
        vocab = {"<pad>": 0, "A": 1}
        id_to = {v: k for k, v in vocab.items()}
        lp = _make_logits(len(vocab), [1])
        with self.assertRaises(ValueError):
            topk_beam_decode(lp, id_to_phoneme=id_to, top_k=0)


class PromptFormatterTests(unittest.TestCase):
    def test_one_best_prompt_contains_phonemes(self):
        p = format_one_best_prompt(["DH", "AH", "K"])
        self.assertIn("<PHONEMES>", p)
        self.assertIn("DH AH K", p)
        self.assertIn("<TEXT>", p)

    def test_topk_prompt_contains_numbered_list_and_scores(self):
        hyps = [
            CTCHypothesis(["DH", "AH", "K"], score=-1.0),
            CTCHypothesis(["DH", "AH", "G"], score=-2.5),
        ]
        p = format_topk_prompt(hyps, include_scores=True, style="list")
        self.assertIn("1. DH AH K", p)
        self.assertIn("2. DH AH G", p)
        self.assertIn("score=-1.000", p)
        # Top-1 should also appear in the legacy <PHONEMES> block.
        self.assertIn("<PHONEMES>\nDH AH K\n</PHONEMES>", p)
        self.assertIn("<PHONEME_HYPOTHESES>", p)

    def test_flat_style_has_no_numbering(self):
        hyps = [CTCHypothesis(["DH", "AH"], score=-0.5)]
        p = format_topk_prompt(hyps, include_scores=False, style="flat")
        self.assertNotIn("1. ", p)

    def test_invalid_style_raises(self):
        with self.assertRaises(ValueError):
            format_topk_prompt([CTCHypothesis(["A"], -0.1)], style="weird")


class DecodeAndFormatTests(unittest.TestCase):
    def test_top_k_one_is_baseline(self):
        vocab = {"<pad>": 0, "A": 1, "B": 2}
        id_to = {v: k for k, v in vocab.items()}
        log_probs = _make_logits(len(vocab), [1, 0, 2])
        hyps, prompts = decode_and_format(
            log_probs, id_to_phoneme=id_to, blank_id=0, top_k=1,
            already_log_probs=True,
        )
        self.assertEqual(len(hyps), 1)
        self.assertEqual(len(hyps[0]), 1)
        self.assertEqual(hyps[0][0].phonemes, ["A", "B"])
        self.assertIn("<PHONEMES>\nA B\n</PHONEMES>", prompts[0])

    def test_top_k_many_uses_topk_prompt(self):
        vocab = {"<pad>": 0, "A": 1, "B": 2}
        id_to = {v: k for k, v in vocab.items()}
        log_probs = _make_logits(len(vocab), [1, 0, 2])
        hyps, prompts = decode_and_format(
            log_probs, id_to_phoneme=id_to, blank_id=0, top_k=3,
            already_log_probs=True,
        )
        # At least 1 hypothesis, and the prompt should include top-k block.
        self.assertGreaterEqual(len(hyps[0]), 1)
        self.assertIn("<PHONEME_HYPOTHESES>", prompts[0])


if __name__ == "__main__":
    unittest.main()
