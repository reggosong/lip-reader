"""Unit tests for the multi-scale temporal encoder."""

from __future__ import annotations

import os
import sys
import unittest

import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from Models.temporal_encoder import MultiScaleTemporalEncoder, build_from_config


class MultiScaleTemporalEncoderTests(unittest.TestCase):
    def test_concat_fusion_shape(self):
        enc = MultiScaleTemporalEncoder(
            in_dim=64, hidden_dim=32,
            branch_kernels=(3, 5, 9),
            context_layers=1, context_heads=4,
            downsample_factor=4, dropout=0.0,
        )
        x = torch.randn(2, 64, 64)
        y = enc(x)
        self.assertEqual(y.shape, (2, 64 // 4, 32))

    def test_sum_fusion_shape(self):
        enc = MultiScaleTemporalEncoder(
            in_dim=64, hidden_dim=32,
            branch_kernels=(3, 5),
            fusion="sum",
            context_layers=0,
            downsample_factor=1, dropout=0.0,
        )
        x = torch.randn(1, 48, 64)
        y = enc(x)
        self.assertEqual(y.shape, (1, 48, 32))

    def test_conformer_context(self):
        enc = MultiScaleTemporalEncoder(
            in_dim=32, hidden_dim=16,
            branch_kernels=(3, 5, 7),
            context_type="conformer",
            context_layers=2, context_heads=2,
            downsample_factor=2, dropout=0.1,
        )
        x = torch.randn(1, 32, 32)
        y = enc(x)
        self.assertEqual(y.shape, (1, 16, 16))

    def test_build_from_config(self):
        cfg = {
            "encoder_type": "multiscale",
            "in_dim": 32,
            "hidden_dim": 16,
            "branch_kernels": [3, 5, 9],
            "context_layers": 1,
            "context_heads": 2,
            "downsample_factor": 2,
        }
        enc = build_from_config(cfg)
        self.assertIsNotNone(enc)
        y = enc(torch.randn(1, 24, 32))
        self.assertEqual(y.shape, (1, 12, 16))

    def test_build_from_config_other_encoder_returns_none(self):
        self.assertIsNone(build_from_config(None))
        self.assertIsNone(build_from_config({"encoder_type": "adapter"}))

    def test_invalid_fusion(self):
        with self.assertRaises(ValueError):
            MultiScaleTemporalEncoder(in_dim=16, hidden_dim=8, fusion="bogus")

    def test_invalid_branch_dilations_length(self):
        with self.assertRaises(ValueError):
            MultiScaleTemporalEncoder(
                in_dim=16, hidden_dim=8,
                branch_kernels=(3, 5),
                branch_dilations=(1, 1, 1),
            )

    def test_heads_must_divide_hidden_dim(self):
        with self.assertRaises(ValueError):
            MultiScaleTemporalEncoder(in_dim=16, hidden_dim=18, context_heads=4)

    def test_assert_ctc_length_raises(self):
        enc = MultiScaleTemporalEncoder(
            in_dim=16, hidden_dim=8,
            downsample_factor=4, context_layers=0,
        )
        # Feasible case: no exception.
        enc.assert_ctc_length(enc.expected_output_length(32), max_target_length=4)
        # Infeasible case: too-aggressive downsampling.
        with self.assertRaises(RuntimeError):
            enc.assert_ctc_length(2, max_target_length=5)

    def test_encoder_is_differentiable(self):
        enc = MultiScaleTemporalEncoder(
            in_dim=16, hidden_dim=8,
            branch_kernels=(3, 5), context_layers=1,
            context_heads=2, downsample_factor=2, dropout=0.0,
        )
        x = torch.randn(1, 16, 16, requires_grad=True)
        y = enc(x)
        y.sum().backward()
        self.assertIsNotNone(x.grad)


if __name__ == "__main__":
    unittest.main()
