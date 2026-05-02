"""Print clean / noisy phoneme examples for debugging.

This script is intentionally dependency-light: it only needs
``pronouncing`` (already a project dep) and stdlib, so it runs on CPU-
only dev machines.

Usage::

    python scripts/debug_phoneme_noise.py \
        --sentences "the cat sat on the mat" "good morning everyone" \
        --config configs/noisy_llm.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from Data.phoneme_noise import PhonemeNoiseAugmenter, build_augmenter_from_config  # noqa: E402
from Data.phoneme_utils import text_to_phoneme_line, phoneme_line_to_tokens  # noqa: E402


DEFAULT_SENTENCES = [
    "the cat sat on the mat",
    "good morning everyone",
    "please open the window",
    "she sells seashells by the seashore",
    "a dog and a bird are friends",
]


def _load_cfg(path: str | None):
    if path is None:
        return {"enabled": True, "seed": 0, "substitute_prob": 0.15,
                "insert_prob": 0.05, "delete_prob": 0.05}
    with open(path, "r") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Print clean / noisy phoneme debug examples.")
    parser.add_argument("--sentences", nargs="+", default=DEFAULT_SENTENCES,
                        help="Sentences to convert and perturb.")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to a noisy-LLM JSON config. If omitted, a default one is used.")
    parser.add_argument("--repeats", type=int, default=2,
                        help="Number of noisy samples to print per sentence.")
    args = parser.parse_args()

    cfg = _load_cfg(args.config)
    augmenter = build_augmenter_from_config(cfg)
    if augmenter is None:
        print("Augmenter disabled in config; enabling for debug run.")
        cfg["enabled"] = True
        augmenter = build_augmenter_from_config(cfg)

    assert augmenter is not None
    print(
        f"Noise: sub={augmenter.substitute_prob} ins={augmenter.insert_prob} "
        f"del={augmenter.delete_prob} seed={cfg.get('seed')}"
    )
    print("=" * 72)

    for sentence in args.sentences:
        clean_line = text_to_phoneme_line(sentence)
        print(f"TEXT : {sentence}")
        print(f"CLEAN: {clean_line}")
        tokens = phoneme_line_to_tokens(clean_line)
        for i in range(args.repeats):
            noisy = augmenter(tokens)
            print(f"NOISY[{i}]: {' '.join(noisy)}")
        print("-" * 72)


if __name__ == "__main__":
    main()
