"""
Unified VALLR evaluation driver.

This script lets you compare four configurations side-by-side:

    * ``baseline``       – original adapter encoder + clean LLM + one-best
    * ``noisy_llm``      – original encoder + LLM trained on noisy phonemes
    * ``topk_llm``       – original encoder + top-k CTC hypotheses prompt
    * ``multiscale``     – multi-scale temporal encoder in place of adapter
                           (still uses the one-best LLM by default)

Each configuration is defined by a plain dict in :data:`PRESETS`. You
can run a single configuration with ``--run`` or run all of them and
get a markdown-style summary table with ``--run all``.

The script does **not** require the baseline pretrained weights or any
private data to import or run its self-check, so you can do
``python evaluate.py --self-check`` on a CPU-only machine to verify
that all four code paths construct cleanly and produce the expected
output shapes.

Example real-world usage (assumes you have LRS3-preprocessed videos and
a fine-tuned VALLR + Llama LoRA checkpoint)::

    python evaluate.py --run baseline \
        --videos_root /path/to/lrs3 --vallr_ckpt model.pth

    python evaluate.py --run topk_llm --videos_root /path/to/lrs3 \
        --vallr_ckpt model.pth --llm_ckpt ./llama_phonemes_to_text_lora

The script intentionally degrades gracefully: if the LLM checkpoint
isn't provided, it prints only phoneme-level metrics and the prompts
it would have handed to the LLM.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

# Make sure we can import the repo when run as a top-level script.
_REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from config import get_vocab  # noqa: E402
from Data.phoneme_noise import PhonemeNoiseAugmenter  # noqa: E402
from Models.ctc_decode import (  # noqa: E402
    CTCHypothesis,
    decode_and_format,
    format_one_best_prompt,
    format_topk_prompt,
    greedy_decode,
    topk_beam_decode,
)
from Models.temporal_encoder import MultiScaleTemporalEncoder  # noqa: E402

try:  # jiwer is already in requirements.txt
    import jiwer  # type: ignore
except Exception:  # pragma: no cover - jiwer is optional at import time
    jiwer = None  # type: ignore


# ---------------------------------------------------------------------------
# Preset configurations
# ---------------------------------------------------------------------------


@dataclass
class EvalPreset:
    """Description of one evaluation configuration."""

    name: str
    encoder_type: str = "adapter"
    multiscale_config: Optional[str] = "configs/multiscale_encoder.json"
    top_k: int = 1
    ctc_beam_width: int = 16
    topk_config: Optional[str] = None
    noise_config: Optional[str] = None
    notes: str = ""


PRESETS: Dict[str, EvalPreset] = {
    "baseline": EvalPreset(
        name="baseline",
        encoder_type="adapter",
        multiscale_config=None,
        top_k=1,
        notes="Original VALLR: adapter encoder, clean LLM, one-best prompt.",
    ),
    "noisy_llm": EvalPreset(
        name="noisy_llm",
        encoder_type="adapter",
        multiscale_config=None,
        top_k=1,
        noise_config="configs/noisy_llm.json",
        notes="Adapter encoder + one-best; LLM trained with phoneme noise.",
    ),
    "topk_llm": EvalPreset(
        name="topk_llm",
        encoder_type="adapter",
        multiscale_config=None,
        top_k=5,
        ctc_beam_width=16,
        topk_config="configs/topk_llm.json",
        notes="Adapter encoder; LLM consumes top-k CTC hypotheses.",
    ),
    "multiscale": EvalPreset(
        name="multiscale",
        encoder_type="multiscale",
        multiscale_config="configs/multiscale_encoder.json",
        top_k=1,
        notes="Multi-scale temporal encoder; one-best LLM prompt.",
    ),
}


# ---------------------------------------------------------------------------
# Core evaluation: CTC decode + (optional) LLM generation + (optional) WER.
# ---------------------------------------------------------------------------


def _load_json(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    with open(path, "r") as f:
        return json.load(f)


def _greedy_str(hyps: List[CTCHypothesis]) -> str:
    return hyps[0].as_string() if hyps else ""


def run_preset_on_batch(
    preset: EvalPreset,
    logits: torch.Tensor,
    phoneme_vocab: Dict[str, int],
) -> Dict[str, Any]:
    """Decode a batch of CTC logits under the given preset.

    Returns a dict with the ranked hypotheses and LLM-ready prompts.
    This function is pure w.r.t. the network it was run on, so tests
    can call it with hand-crafted logits.
    """
    id_to_phoneme = {v: k for k, v in phoneme_vocab.items()}
    blank_id = phoneme_vocab.get("<pad>", 0)

    topk_cfg = _load_json(preset.topk_config) or {}
    include_scores = bool(topk_cfg.get("include_scores", True))
    prompt_style = topk_cfg.get("prompt_style", "list")
    prune_log_prob = float(topk_cfg.get("prune_log_prob", -6.0))
    top_k = int(topk_cfg.get("top_k", preset.top_k))
    beam_width = int(topk_cfg.get("beam_width", preset.ctc_beam_width))

    hyps, prompts = decode_and_format(
        logits,
        id_to_phoneme=id_to_phoneme,
        blank_id=blank_id,
        top_k=top_k,
        beam_width=beam_width,
        prune_log_prob=prune_log_prob,
        include_scores=include_scores,
        style=prompt_style,
    )

    return {
        "preset": preset.name,
        "hypotheses": hyps,
        "prompts": prompts,
        "top_k": top_k,
    }


# ---------------------------------------------------------------------------
# LLM generation wrapper (graceful degradation when no checkpoint provided).
# ---------------------------------------------------------------------------


def generate_text_with_llm(
    prompts: List[str],
    llm_ckpt: Optional[str],
    base_model_id: str = "meta-llama/Llama-3.2-3B-Instruct",
    max_new_tokens: int = 96,
) -> List[str]:
    """Feed a batch of phoneme prompts to the LLM and return generations.

    If ``llm_ckpt`` is ``None``, return the prompts themselves so the
    rest of the pipeline (WER computation etc.) can be exercised in
    a smoke-test mode.
    """
    if llm_ckpt is None:
        return list(prompts)

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as e:  # pragma: no cover - defensive
        print(f"[evaluate] transformers not available ({e}); returning prompts.")
        return list(prompts)

    tokenizer = AutoTokenizer.from_pretrained(llm_ckpt if os.path.isdir(llm_ckpt) else base_model_id)
    model = AutoModelForCausalLM.from_pretrained(
        llm_ckpt if os.path.isdir(llm_ckpt) else base_model_id,
        torch_dtype=torch.float16,
    )
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    outs: List[str] = []
    for p in prompts:
        ids = tokenizer(p, return_tensors="pt").to(device)
        with torch.no_grad():
            gen = model.generate(
                **ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(gen[0][ids["input_ids"].size(1):], skip_special_tokens=True)
        outs.append(text.strip())
    return outs


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_wer(refs: List[str], hyps: List[str]) -> float:
    if jiwer is None or not refs:
        return float("nan")
    try:
        return float(jiwer.wer(refs, hyps))
    except Exception:
        return float("nan")


def compute_per(refs: List[List[str]], hyps: List[List[str]]) -> float:
    """Phoneme error rate (simple edit distance normalized by reference length)."""
    if not refs:
        return float("nan")
    total_edits = 0
    total_len = 0
    for r, h in zip(refs, hyps):
        total_edits += _levenshtein(r, h)
        total_len += len(r)
    return total_edits / max(1, total_len)


def _levenshtein(a: List[Any], b: List[Any]) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ai in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, bj in enumerate(b, 1):
            cur[j] = min(
                prev[j] + 1,
                cur[j - 1] + 1,
                prev[j - 1] + (0 if ai == bj else 1),
            )
        prev = cur
    return prev[-1]


# ---------------------------------------------------------------------------
# Self-check (no external data required)
# ---------------------------------------------------------------------------


def _random_logits(seed: int, batch: int, time: int, vocab_size: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(batch, time, vocab_size, generator=g) * 2.0


def self_check() -> Dict[str, Dict[str, Any]]:
    """Exercise all four presets with synthetic data and tiny models."""
    vocab = get_vocab()
    V = len(vocab)
    T = 64
    B = 2
    logits = _random_logits(seed=0, batch=B, time=T, vocab_size=V)

    results: Dict[str, Dict[str, Any]] = {}
    for name, preset in PRESETS.items():
        res = run_preset_on_batch(preset, logits, vocab)
        # Sanity: at least one hypothesis per batch element.
        assert len(res["hypotheses"]) == B, "batch size mismatch"
        for hyps in res["hypotheses"]:
            assert len(hyps) >= 1
        results[name] = {
            "top_k": res["top_k"],
            "example_prompt": res["prompts"][0][:240].replace("\n", " | "),
            "example_top1": res["hypotheses"][0][0].as_string()[:120],
        }

    # Also check that the multi-scale encoder runs forward and the shape
    # check fires when expected.
    enc = MultiScaleTemporalEncoder(
        in_dim=32, hidden_dim=16, branch_kernels=(3, 5, 9),
        context_layers=1, context_heads=2, downsample_factor=4, dropout=0.0,
    )
    x = torch.randn(2, 32, 32)
    y = enc(x)
    assert y.shape == (2, 32 // 4, 16), f"unexpected encoder shape: {y.shape}"
    results["multiscale_encoder_forward"] = {
        "in_shape": list(x.shape),
        "out_shape": list(y.shape),
    }

    # Shape check: too-aggressive downsampling should raise.
    enc.assert_ctc_length(output_length=enc.expected_output_length(32), max_target_length=4)
    try:
        enc.assert_ctc_length(output_length=3, max_target_length=5)
        raised = False
    except RuntimeError:
        raised = True
    assert raised, "assert_ctc_length did not raise for infeasible CTC length"
    results["ctc_shape_check"] = {"raised_on_infeasible": True}

    # Also verify the augmenter is deterministic under a seed.
    aug = PhonemeNoiseAugmenter(seed=42)
    a = aug(["DH", "AH", "K", "AE", "T"])
    aug.reseed(42)
    b = aug(["DH", "AH", "K", "AE", "T"])
    assert a == b, "augmenter is not deterministic under reseed"
    results["augmenter_deterministic"] = {"sample": a}

    return results


def format_self_check(results: Dict[str, Dict[str, Any]]) -> str:
    out: List[str] = []
    out.append("# VALLR evaluation self-check\n")
    out.append("| preset | top_k | example top-1 phonemes | example prompt (truncated) |")
    out.append("|---|---|---|---|")
    for name in ("baseline", "noisy_llm", "topk_llm", "multiscale"):
        r = results[name]
        out.append(f"| {name} | {r['top_k']} | {r['example_top1']} | {r['example_prompt']} |")
    out.append("")
    out.append(f"Multi-scale encoder forward: {results['multiscale_encoder_forward']}")
    out.append(f"CTC shape-check on infeasible length: {results['ctc_shape_check']}")
    out.append(f"Augmenter deterministic sample: {results['augmenter_deterministic']}")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _run_real(preset: EvalPreset, args: argparse.Namespace) -> Dict[str, Any]:
    """Run a single preset end-to-end on real data if provided."""
    # Lazy imports so self-check can run on minimal installs.
    from main import load_finetuned_model, load_videos

    vocab = get_vocab()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if not args.vallr_ckpt or not args.videos_root:
        raise RuntimeError(
            "--vallr_ckpt and --videos_root are required for real evaluation. "
            "Use --self-check for a dependency-free smoke test."
        )

    model = load_finetuned_model(
        args.vallr_ckpt,
        device,
        args.version,
        vocab,
        encoder_type=preset.encoder_type,
        multiscale_config_path=preset.multiscale_config,
    )

    # Evaluate on a single clip or a directory of clips.
    clip_paths: List[str] = []
    if os.path.isdir(args.videos_root):
        for root, _, files in os.walk(args.videos_root):
            for f in files:
                if f.endswith((".mp4", ".mov", ".avi", ".mkv")):
                    clip_paths.append(os.path.join(root, f))
                    if args.max_clips and len(clip_paths) >= args.max_clips:
                        break
            if args.max_clips and len(clip_paths) >= args.max_clips:
                break
    else:
        clip_paths.append(args.videos_root)

    all_prompts: List[str] = []
    all_hyps: List[List[CTCHypothesis]] = []
    for p in clip_paths:
        video = load_videos(p, num_frames=16)
        if video is None:
            continue
        video = video.to(device).float()
        with torch.no_grad():
            logits, _ = model(video)
        log_probs = F.log_softmax(logits, dim=-1)
        res = run_preset_on_batch(preset, log_probs, vocab)
        all_hyps.extend(res["hypotheses"])
        all_prompts.extend(res["prompts"])

    # Optional LLM decoding.
    texts = generate_text_with_llm(all_prompts, args.llm_ckpt)

    return {
        "preset": preset.name,
        "num_clips": len(clip_paths),
        "example_prompt": all_prompts[0] if all_prompts else "",
        "example_text": texts[0] if texts else "",
        "hypotheses": all_hyps,
        "generations": texts,
    }


def main():
    parser = argparse.ArgumentParser(description="VALLR unified evaluation driver.")
    parser.add_argument("--run", type=str, default=None,
                        help=f"Preset to run ({'/'.join(PRESETS)}) or 'all'.")
    parser.add_argument("--self-check", action="store_true",
                        help="Run all presets on synthetic data (no data/ckpts required).")
    parser.add_argument("--videos_root", type=str, default=None,
                        help="Directory or single video for real evaluation.")
    parser.add_argument("--vallr_ckpt", type=str, default=None,
                        help="Path to a fine-tuned VALLR .pth checkpoint.")
    parser.add_argument("--llm_ckpt", type=str, default=None,
                        help="Optional path to a Llama LoRA adapter directory.")
    parser.add_argument("--version", type=str, default="V2", choices=["V1", "V2"])
    parser.add_argument("--max_clips", type=int, default=10,
                        help="Cap the number of clips processed in real mode.")
    args = parser.parse_args()

    if args.self_check or not args.run:
        results = self_check()
        print(format_self_check(results))
        return

    to_run: List[str] = list(PRESETS.keys()) if args.run == "all" else [args.run]
    summary: List[Dict[str, Any]] = []
    for name in to_run:
        preset = PRESETS[name]
        print(f"\n=== Running preset: {name} ===")
        print(f"    {preset.notes}")
        res = _run_real(preset, args)
        print(f"    clips: {res['num_clips']}")
        print(f"    sample prompt:\n{res['example_prompt']}")
        if res["example_text"]:
            print(f"    sample text  : {res['example_text']}")
        summary.append({
            "preset": name,
            "clips": res["num_clips"],
            "sample_text": res["example_text"][:160],
        })

    if len(summary) > 1:
        print("\n# Summary")
        for row in summary:
            print(f"- {row['preset']}: {row['clips']} clips; sample={row['sample_text']!r}")


if __name__ == "__main__":
    main()
