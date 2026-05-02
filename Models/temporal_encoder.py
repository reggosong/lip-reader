"""
Multi-scale temporal encoder for VALLR (Task 3).

The encoder consumes a ViT token sequence of shape ``(B, N, D_in)``
(where ``N`` plays the role of time in the VALLR backbone) and
produces a sequence ``(B, N_out, D_out)`` suitable for the CTC head.
It is drop-in compatible with :class:`Models.ML_VALLR.VALLR` – when
``encoder_type == "multiscale"`` is requested, VALLR instantiates this
module in place of the original downsampling + linear adapter.

Architecture
------------
1. **Multi-scale convolutional branches.** Three (or more) parallel
   1-D convolutions with different kernel sizes capture short, medium
   and long temporal contexts. Each branch outputs ``hidden_dim``
   channels.
2. **Fusion.** Branch outputs are fused via either concatenation
   followed by a 1x1 projection (``"concat"``) or element-wise sum
   (``"sum"``).
3. **Optional temporal context.** A Transformer encoder (standard
   self-attention) or a light Conformer-style block (self-attention +
   depthwise conv FFN) is applied over the fused sequence. The number
   of layers is configurable and may be ``0``.
4. **Downsampling.** A final strided 1-D convolution downsamples the
   sequence by ``downsample_factor`` to keep CTC output lengths in
   the same ball-park as the baseline adapter path.

The module is deliberately config-driven (via :func:`build_from_config`)
so callers can switch from the baseline adapter to the multi-scale
encoder purely by supplying a JSON config.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helper blocks
# ---------------------------------------------------------------------------


class _Conv1dBranch(nn.Module):
    """A single temporal Conv1d branch with BN + GELU."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        kernel_size: int,
        dilation: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        # "Same" padding for a causal-free, length-preserving branch.
        padding = ((kernel_size - 1) * dilation) // 2
        self.conv = nn.Conv1d(
            in_channels=in_dim,
            out_channels=out_dim,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
        )
        self.bn = nn.BatchNorm1d(out_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, D, T)
        h = self.conv(x)
        # Align length in case of small off-by-one from the chosen padding.
        if h.size(-1) != x.size(-1):
            h = h[..., : x.size(-1)] if h.size(-1) > x.size(-1) else F.pad(h, (0, x.size(-1) - h.size(-1)))
        return self.drop(self.act(self.bn(h)))


class _ConformerFFN(nn.Module):
    """A tiny Conformer-style convolutional FFN block."""

    def __init__(self, dim: int, expansion: int = 2, kernel_size: int = 7, dropout: float = 0.0) -> None:
        super().__init__()
        inner = dim * expansion
        self.norm = nn.LayerNorm(dim)
        self.pointwise_in = nn.Conv1d(dim, inner, kernel_size=1)
        self.depthwise = nn.Conv1d(
            inner, inner, kernel_size=kernel_size,
            padding=kernel_size // 2, groups=inner,
        )
        self.act = nn.GELU()
        self.pointwise_out = nn.Conv1d(inner, dim, kernel_size=1)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, D)
        h = self.norm(x).transpose(1, 2)        # (B, D, T)
        h = self.pointwise_in(h)
        h = self.act(self.depthwise(h))
        h = self.pointwise_out(h).transpose(1, 2)  # (B, T, D)
        return x + self.drop(h)


class _ConformerBlock(nn.Module):
    """A minimal Conformer block: MHSA + Conv FFN + FFN."""

    def __init__(self, dim: int, heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=heads, dropout=dropout, batch_first=True)
        self.attn_drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv_ffn = _ConformerFFN(dim, dropout=dropout)
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.attn_norm(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.attn_drop(a)
        x = self.conv_ffn(x)
        x = x + self.mlp(self.mlp_norm(x))
        return x


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


class MultiScaleTemporalEncoder(nn.Module):
    """Multi-scale temporal encoder.

    Parameters
    ----------
    in_dim:
        Input feature dimension (ViT hidden size, typically 768).
    hidden_dim:
        Hidden channel dimension produced by each branch and the
        fusion layer (also the model dim of the context layers).
    branch_kernels:
        Kernel sizes for the parallel branches. ``[3, 5, 9]`` by default
        corresponds to short / medium / long receptive fields.
    branch_dilations:
        Optional matching list of dilations (same length as kernels).
    fusion:
        ``"concat"`` (default) or ``"sum"``.
    context_layers:
        Number of self-attention context layers applied after fusion.
        ``0`` disables this stage.
    context_type:
        ``"transformer"`` (default) uses PyTorch's standard encoder, or
        ``"conformer"`` uses the lightweight Conformer-style block
        defined in this file.
    context_heads:
        Number of attention heads for the context layers.
    downsample_factor:
        Stride of the final 1-D conv. ``1`` disables downsampling.
    dropout:
        Dropout used throughout.
    """

    def __init__(
        self,
        in_dim: int = 768,
        hidden_dim: int = 256,
        branch_kernels: Sequence[int] = (3, 5, 9),
        branch_dilations: Optional[Sequence[int]] = None,
        fusion: str = "concat",
        context_layers: int = 2,
        context_heads: int = 4,
        context_type: str = "transformer",
        downsample_factor: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if len(branch_kernels) < 2:
            raise ValueError("MultiScaleTemporalEncoder expects at least two branches.")
        if branch_dilations is None:
            branch_dilations = [1] * len(branch_kernels)
        if len(branch_dilations) != len(branch_kernels):
            raise ValueError("branch_dilations must match branch_kernels in length.")
        if fusion not in ("concat", "sum"):
            raise ValueError(f"Unknown fusion mode: {fusion}")
        if context_type not in ("transformer", "conformer", "none"):
            raise ValueError(f"Unknown context_type: {context_type}")
        if downsample_factor < 1:
            raise ValueError("downsample_factor must be >= 1.")
        if hidden_dim % max(1, context_heads) != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by context_heads ({context_heads})."
            )

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.fusion = fusion
        self.context_type = context_type
        self.downsample_factor = downsample_factor

        self.branches = nn.ModuleList([
            _Conv1dBranch(in_dim, hidden_dim, k, d, dropout=dropout)
            for k, d in zip(branch_kernels, branch_dilations)
        ])

        if fusion == "concat":
            self.fuse_proj = nn.Conv1d(hidden_dim * len(branch_kernels), hidden_dim, kernel_size=1)
        else:
            self.fuse_proj = nn.Identity()
        self.fuse_norm = nn.LayerNorm(hidden_dim)

        # Context stack.
        if context_layers > 0 and context_type == "transformer":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=context_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.context = nn.TransformerEncoder(encoder_layer, num_layers=context_layers)
        elif context_layers > 0 and context_type == "conformer":
            self.context = nn.ModuleList([
                _ConformerBlock(hidden_dim, heads=context_heads, dropout=dropout)
                for _ in range(context_layers)
            ])
        else:
            self.context = None

        # Final downsampling.
        if downsample_factor > 1:
            self.downsample = nn.Conv1d(
                hidden_dim, hidden_dim,
                kernel_size=downsample_factor,
                stride=downsample_factor,
                padding=0,
            )
        else:
            self.downsample = nn.Identity()

        self.out_norm = nn.LayerNorm(hidden_dim)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    @property
    def out_dim(self) -> int:
        return self.hidden_dim

    def expected_output_length(self, input_length: int) -> int:
        """Return the sequence length produced for a given input length."""
        if self.downsample_factor <= 1:
            return input_length
        return input_length // self.downsample_factor

    def assert_ctc_length(self, output_length: int, max_target_length: int) -> None:
        """Raise an informative error if CTC can't fit the targets.

        CTC requires ``T_out >= max_target_length`` (strictly, with
        repeat collapsing, ``T_out`` must be at least as long as the
        longest target after insertion of the required blanks).
        """
        if output_length < max_target_length:
            raise RuntimeError(
                "MultiScaleTemporalEncoder output is too short for CTC: "
                f"T_out={output_length} < max_target_length={max_target_length}. "
                "Reduce `downsample_factor` or increase the number of input frames."
            )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Encode a token sequence.

        Parameters
        ----------
        tokens:
            Tensor of shape ``(B, N, D_in)``.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(B, N_out, hidden_dim)``.
        """
        if tokens.ndim != 3:
            raise ValueError(f"Expected (B, N, D), got {tuple(tokens.shape)}")
        if tokens.size(-1) != self.in_dim:
            raise ValueError(
                f"Input channel dim {tokens.size(-1)} does not match in_dim={self.in_dim}"
            )

        x = tokens.transpose(1, 2)                  # (B, D_in, N)

        branch_outs: List[torch.Tensor] = [b(x) for b in self.branches]
        if self.fusion == "concat":
            fused = torch.cat(branch_outs, dim=1)   # (B, hidden*K, N)
            fused = self.fuse_proj(fused)           # (B, hidden, N)
        else:
            fused = torch.stack(branch_outs, dim=0).sum(dim=0)  # (B, hidden, N)

        fused = fused.transpose(1, 2)               # (B, N, hidden)
        fused = self.fuse_norm(fused)

        if self.context is not None:
            if self.context_type == "transformer":
                fused = self.context(fused)
            else:  # conformer
                for blk in self.context:
                    fused = blk(fused)

        # Downsample.
        y = fused.transpose(1, 2)                   # (B, hidden, N)
        y = self.downsample(y)                      # (B, hidden, N_out)
        y = y.transpose(1, 2)                       # (B, N_out, hidden)
        y = self.out_norm(y)
        return y


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------


def build_from_config(cfg: Optional[Dict]) -> Optional[MultiScaleTemporalEncoder]:
    """Build a :class:`MultiScaleTemporalEncoder` from a config dict.

    Recognised keys (all optional):
        in_dim, hidden_dim, branch_kernels, branch_dilations, fusion,
        context_layers, context_heads, context_type,
        downsample_factor, dropout.

    Returns ``None`` when ``cfg`` is ``None`` or when
    ``cfg["encoder_type"] != "multiscale"``.
    """
    if cfg is None:
        return None
    if cfg.get("encoder_type", "multiscale") != "multiscale":
        return None
    return MultiScaleTemporalEncoder(
        in_dim=int(cfg.get("in_dim", 768)),
        hidden_dim=int(cfg.get("hidden_dim", 256)),
        branch_kernels=tuple(cfg.get("branch_kernels", (3, 5, 9))),
        branch_dilations=tuple(cfg["branch_dilations"]) if "branch_dilations" in cfg else None,
        fusion=str(cfg.get("fusion", "concat")),
        context_layers=int(cfg.get("context_layers", 2)),
        context_heads=int(cfg.get("context_heads", 4)),
        context_type=str(cfg.get("context_type", "transformer")),
        downsample_factor=int(cfg.get("downsample_factor", 4)),
        dropout=float(cfg.get("dropout", 0.1)),
    )
