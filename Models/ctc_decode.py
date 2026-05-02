"""
CTC decoding utilities and top-k hypothesis prompt formatters.

This module provides:

* :func:`greedy_decode` – the existing one-best path used by the
  baseline VALLR inference (collapse repeats, drop blanks).
* :func:`topk_beam_decode` – a prefix-beam CTC decoder that returns the
  top-K phoneme hypotheses with their log-probability scores.
* :func:`format_one_best_prompt` / :func:`format_topk_prompt` – prompt
  formatters consumed by the phoneme-to-text LLM. The one-best
  formatter reproduces the original VALLR prompt, so existing LLM
  checkpoints continue to work.

The module is intentionally free of heavy dependencies (no ctcdecode,
no pyctcdecode, no wandb) so it can be unit-tested cheaply and used
from both training and eval scripts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class CTCHypothesis:
    """A single decoded phoneme hypothesis.

    Attributes
    ----------
    phonemes:
        List of phoneme strings, blanks already removed and adjacent
        duplicates collapsed.
    score:
        Log-probability of this hypothesis under the CTC model.
    """

    phonemes: List[str]
    score: float

    def as_string(self) -> str:
        return " ".join(self.phonemes)


# ---------------------------------------------------------------------------
# One-best (greedy) decoder
# ---------------------------------------------------------------------------


def greedy_decode(
    log_probs: torch.Tensor,
    id_to_phoneme: Dict[int, str],
    blank_id: int = 0,
) -> List[CTCHypothesis]:
    """Greedy CTC decoding.

    Parameters
    ----------
    log_probs:
        Tensor of shape ``(B, T, V)`` or ``(T, V)`` containing log-probs.
    id_to_phoneme:
        Mapping from class id to phoneme string. Missing ids are dropped.
    blank_id:
        The CTC blank index (default ``0``).

    Returns
    -------
    list[CTCHypothesis]
        One hypothesis per batch element. The score is the sum of the
        log-probs of the argmax path (not of the collapsed sequence).
    """
    if log_probs.ndim == 2:
        log_probs = log_probs.unsqueeze(0)
    if log_probs.ndim != 3:
        raise ValueError(f"Expected 2D or 3D log_probs, got {tuple(log_probs.shape)}")

    results: List[CTCHypothesis] = []
    top_vals, top_idx = log_probs.max(dim=-1)  # (B, T)
    for b in range(log_probs.size(0)):
        ids = top_idx[b].tolist()
        path_score = float(top_vals[b].sum().item())

        phonemes: List[str] = []
        prev = None
        for i in ids:
            if i == blank_id:
                prev = None
                continue
            if i == prev:
                continue
            ph = id_to_phoneme.get(i)
            if ph is None:
                prev = i
                continue
            phonemes.append(ph)
            prev = i

        results.append(CTCHypothesis(phonemes=phonemes, score=path_score))
    return results


# ---------------------------------------------------------------------------
# Top-K prefix beam search
# ---------------------------------------------------------------------------


def _logsumexp2(a: float, b: float) -> float:
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


def _prefix_beam_search_single(
    log_probs: torch.Tensor,  # (T, V)
    id_to_phoneme: Dict[int, str],
    blank_id: int,
    beam_width: int,
    top_k: int,
    prune_log_prob: float,
) -> List[CTCHypothesis]:
    """Run prefix-beam CTC on a single (T, V) log-prob matrix.

    Implements the standard prefix beam search (Graves / Hannun) where
    each prefix tracks two probabilities: one for paths ending in
    blank (``pb``) and one for paths ending in a non-blank symbol
    (``pnb``).
    """
    T, V = log_probs.shape
    if T == 0:
        return [CTCHypothesis(phonemes=[], score=0.0)]

    NEG_INF = -math.inf
    # Beam entries: prefix (tuple of ids) -> (log_pb, log_pnb)
    beam: Dict[Tuple[int, ...], Tuple[float, float]] = {(): (0.0, NEG_INF)}

    log_probs_cpu = log_probs.detach().cpu()

    for t in range(T):
        row = log_probs_cpu[t]
        # Optional pruning: only consider tokens above `prune_log_prob`.
        candidate_ids: Iterable[int]
        if prune_log_prob > NEG_INF:
            mask = row > prune_log_prob
            if mask.any():
                candidate_ids = torch.nonzero(mask, as_tuple=False).squeeze(-1).tolist()
            else:
                # Fall back to the top tokens if pruning killed everything.
                candidate_ids = torch.topk(row, k=min(beam_width, V)).indices.tolist()
        else:
            candidate_ids = range(V)
        row_list = row.tolist()

        new_beam: Dict[Tuple[int, ...], Tuple[float, float]] = {}

        def add(prefix: Tuple[int, ...], pb: float, pnb: float) -> None:
            cur_pb, cur_pnb = new_beam.get(prefix, (NEG_INF, NEG_INF))
            new_beam[prefix] = (_logsumexp2(cur_pb, pb), _logsumexp2(cur_pnb, pnb))

        for prefix, (log_pb, log_pnb) in beam.items():
            last = prefix[-1] if prefix else None

            # Blank extension: stays on the same prefix.
            lp_blank = row_list[blank_id]
            add(prefix, _logsumexp2(log_pb + lp_blank, log_pnb + lp_blank), NEG_INF)

            for c in candidate_ids:
                if c == blank_id:
                    continue
                lp_c = row_list[c]

                if c == last:
                    # Two cases for same symbol:
                    # - extend path ending in blank with c -> new non-blank
                    # - extend path ending in non-blank (same symbol, no new char) -> merged
                    add(prefix + (c,), NEG_INF, log_pb + lp_c)
                    add(prefix,         NEG_INF, log_pnb + lp_c)
                else:
                    new_prefix = prefix + (c,)
                    merged = _logsumexp2(log_pb, log_pnb) + lp_c
                    add(new_prefix, NEG_INF, merged)

        # Prune to top `beam_width` prefixes by total log-prob.
        scored = [
            (pref, _logsumexp2(pb, pnb), pb, pnb)
            for pref, (pb, pnb) in new_beam.items()
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        scored = scored[:beam_width]
        beam = {pref: (pb, pnb) for pref, _, pb, pnb in scored}

    # Final ranking.
    finals = [
        (pref, _logsumexp2(pb, pnb))
        for pref, (pb, pnb) in beam.items()
    ]
    finals.sort(key=lambda x: x[1], reverse=True)
    finals = finals[: max(1, top_k)]

    hyps: List[CTCHypothesis] = []
    for prefix, score in finals:
        phonemes = [id_to_phoneme[i] for i in prefix if i in id_to_phoneme]
        hyps.append(CTCHypothesis(phonemes=phonemes, score=float(score)))
    return hyps


def topk_beam_decode(
    log_probs: torch.Tensor,
    id_to_phoneme: Dict[int, str],
    blank_id: int = 0,
    beam_width: int = 16,
    top_k: int = 5,
    prune_log_prob: float = -6.0,
) -> List[List[CTCHypothesis]]:
    """Run prefix-beam CTC decoding and return the top-K hypotheses.

    Parameters
    ----------
    log_probs:
        Tensor shaped ``(B, T, V)`` or ``(T, V)`` with *log*-probabilities.
    id_to_phoneme:
        Mapping from class index to phoneme string.
    blank_id:
        Index of the CTC blank token.
    beam_width:
        Number of beams kept per time step.
    top_k:
        Number of hypotheses returned for each batch element.
    prune_log_prob:
        Tokens with per-frame log-prob below this threshold are skipped.
        Set to ``-inf`` to disable pruning.

    Returns
    -------
    list[list[CTCHypothesis]]
        Outer list is batch, inner list is hypotheses sorted by score
        (highest first).
    """
    if log_probs.ndim == 2:
        log_probs = log_probs.unsqueeze(0)
    if log_probs.ndim != 3:
        raise ValueError(f"Expected 2D or 3D log_probs, got {tuple(log_probs.shape)}")
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")
    if beam_width < top_k:
        beam_width = top_k

    B = log_probs.size(0)
    out: List[List[CTCHypothesis]] = []
    for b in range(B):
        hyps = _prefix_beam_search_single(
            log_probs[b],
            id_to_phoneme=id_to_phoneme,
            blank_id=blank_id,
            beam_width=beam_width,
            top_k=top_k,
            prune_log_prob=prune_log_prob,
        )
        out.append(hyps)
    return out


# ---------------------------------------------------------------------------
# Prompt formatters
# ---------------------------------------------------------------------------

# Tags are kept in sync with Models/Llama.py.
_TAG_START      = "<S2S>"
_TAG_PH_OPEN    = "<PHONEMES>"
_TAG_PH_CLOSE   = "</PHONEMES>"
_TAG_TOPK_OPEN  = "<PHONEME_HYPOTHESES>"
_TAG_TOPK_CLOSE = "</PHONEME_HYPOTHESES>"
_TAG_TEXT       = "<TEXT>"


def format_one_best_prompt(phonemes: Sequence[str]) -> str:
    """Original one-best VALLR prompt. Preserved for backward compatibility."""
    line = " ".join(phonemes)
    return (
        f"{_TAG_START}\n"
        f"{_TAG_PH_OPEN}\n{line}\n{_TAG_PH_CLOSE}\n"
        f"{_TAG_TEXT}\n"
    )


def format_topk_prompt(
    hypotheses: Sequence[CTCHypothesis],
    *,
    include_scores: bool = True,
    style: str = "list",
) -> str:
    """Format a list of top-k phoneme hypotheses into an LLM prompt.

    Parameters
    ----------
    hypotheses:
        Already-ranked list of :class:`CTCHypothesis` (best first).
    include_scores:
        When ``True`` each line is annotated with its log-prob score.
    style:
        ``"list"`` – numbered hypotheses, one per line (default).
        ``"flat"`` – same but without the numbering, for smaller LLMs.

    The top-1 line is also written under the standard ``<PHONEMES>``
    tag so LLM checkpoints trained on the original prompt still work.
    """
    if not hypotheses:
        return format_one_best_prompt([])

    top1 = hypotheses[0]

    if style not in ("list", "flat"):
        raise ValueError(f"Unknown prompt style: {style}")

    lines: List[str] = []
    for i, h in enumerate(hypotheses):
        phon_line = h.as_string()
        if style == "list":
            prefix = f"{i + 1}. "
        else:
            prefix = ""
        if include_scores:
            lines.append(f"{prefix}{phon_line}\t(score={h.score:.3f})")
        else:
            lines.append(f"{prefix}{phon_line}")
    topk_block = "\n".join(lines)

    return (
        f"{_TAG_START}\n"
        f"{_TAG_PH_OPEN}\n{top1.as_string()}\n{_TAG_PH_CLOSE}\n"
        f"{_TAG_TOPK_OPEN}\n{topk_block}\n{_TAG_TOPK_CLOSE}\n"
        f"{_TAG_TEXT}\n"
    )


def format_prompt_for_llm(
    hypotheses: Sequence[CTCHypothesis],
    *,
    top_k: int = 1,
    include_scores: bool = True,
    style: str = "list",
) -> str:
    """Uniform entry point used by the evaluation script.

    ``top_k == 1`` dispatches to :func:`format_one_best_prompt` so the
    baseline path is byte-identical to the original VALLR prompt.
    """
    if top_k <= 1 or len(hypotheses) <= 1:
        first = hypotheses[0].phonemes if hypotheses else []
        return format_one_best_prompt(first)
    return format_topk_prompt(
        hypotheses[:top_k], include_scores=include_scores, style=style
    )


# ---------------------------------------------------------------------------
# Convenience: decode a full batch and format prompts in one call.
# ---------------------------------------------------------------------------


def decode_and_format(
    logits: torch.Tensor,
    id_to_phoneme: Dict[int, str],
    *,
    blank_id: int = 0,
    top_k: int = 1,
    beam_width: int = 16,
    prune_log_prob: float = -6.0,
    include_scores: bool = True,
    style: str = "list",
    already_log_probs: bool = False,
) -> Tuple[List[List[CTCHypothesis]], List[str]]:
    """Decode CTC logits and produce LLM prompts.

    Returns ``(per_batch_hypotheses, per_batch_prompts)``.
    """
    log_probs = logits if already_log_probs else F.log_softmax(logits, dim=-1)

    if top_k <= 1:
        greedy = greedy_decode(log_probs, id_to_phoneme=id_to_phoneme, blank_id=blank_id)
        hyps = [[h] for h in greedy]
    else:
        hyps = topk_beam_decode(
            log_probs,
            id_to_phoneme=id_to_phoneme,
            blank_id=blank_id,
            beam_width=beam_width,
            top_k=top_k,
            prune_log_prob=prune_log_prob,
        )

    prompts = [
        format_prompt_for_llm(h_list, top_k=top_k, include_scores=include_scores, style=style)
        for h_list in hyps
    ]
    return hyps, prompts
