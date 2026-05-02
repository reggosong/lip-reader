"""
Noise-aware phoneme augmentation utilities for VALLR.

This module provides ``PhonemeNoiseAugmenter`` and a configurable visual
phoneme confusion map. The augmenter simulates the kinds of errors a
video-only CTC phoneme model tends to produce (visemic confusions,
insertions and deletions) so that the downstream phoneme-to-text LLM can
be fine-tuned to be robust to those errors.

The original, clean training path is preserved: construct an augmenter
with ``substitute_prob=insert_prob=delete_prob=0`` (or simply do not use
it) and the output is identical to the input.

Usage
-----
>>> aug = PhonemeNoiseAugmenter(seed=0)
>>> aug(["DH", "AH", "K", "AE", "T"])            # may return e.g.
["DH", "AH", "G", "AE", "T"]

The module is pure-Python (only depends on ``random``) so it can be used
from anywhere in the repo, including inside data loaders.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Default visual confusion map (ARPAbet, stress-stripped).
#
# Each entry maps a phoneme to a list of ``(candidate, weight)`` pairs.
# Weights are unnormalised: the probability of picking a given candidate
# is ``weight / sum(weights)`` *conditional on* a substitution being
# performed for that source phoneme. Phonemes without an entry fall back
# to the "global" list (all other phonemes, uniform).
#
# The weights are based on standard viseme groupings (bilabials, labio-
# dentals, interdentals, alveolars, sibilants, velars, liquids, rounded /
# unrounded vowels, etc.) reported in the lip-reading literature.
# They are intentionally rough; callers can override the map entirely.
# ---------------------------------------------------------------------------

DEFAULT_VISUAL_CONFUSION_MAP: Dict[str, List[Tuple[str, float]]] = {
    # Bilabials: B, P, M look nearly identical on the lips.
    "B":  [("P", 0.45), ("M", 0.30)],
    "P":  [("B", 0.45), ("M", 0.30)],
    "M":  [("B", 0.35), ("P", 0.35)],
    # Labiodentals.
    "F":  [("V", 0.55)],
    "V":  [("F", 0.55)],
    # Interdentals.
    "TH": [("DH", 0.45), ("F", 0.15), ("S", 0.10)],
    "DH": [("TH", 0.45), ("V", 0.15), ("Z", 0.10)],
    # Alveolar stops / nasals.
    "T":  [("D", 0.35), ("N", 0.15)],
    "D":  [("T", 0.35), ("N", 0.15)],
    "N":  [("T", 0.15), ("D", 0.15), ("L", 0.15), ("NG", 0.10)],
    "L":  [("R", 0.20), ("N", 0.15)],
    # Sibilants & post-alveolars.
    "S":  [("Z", 0.35), ("SH", 0.10)],
    "Z":  [("S", 0.35), ("ZH", 0.10)],
    "SH": [("ZH", 0.45), ("CH", 0.15), ("S", 0.10)],
    "ZH": [("SH", 0.45), ("JH", 0.15), ("Z", 0.10)],
    # Affricates.
    "CH": [("SH", 0.35), ("JH", 0.20), ("T", 0.10)],
    "JH": [("ZH", 0.35), ("CH", 0.20), ("D", 0.10)],
    # Velars.
    "K":  [("G", 0.40), ("T", 0.10)],
    "G":  [("K", 0.40), ("D", 0.10)],
    "NG": [("N", 0.45)],
    # Approximants.
    "R":  [("W", 0.20), ("L", 0.20), ("ER", 0.15)],
    "W":  [("R", 0.25), ("V", 0.10), ("UW", 0.10)],
    "Y":  [("IY", 0.25)],
    "HH": [("AH", 0.25)],
    # High front vowels.
    "IY": [("IH", 0.45), ("EY", 0.20)],
    "IH": [("IY", 0.45), ("EH", 0.20)],
    # Mid front vowels.
    "EY": [("EH", 0.35), ("IY", 0.15), ("AY", 0.10)],
    "EH": [("AE", 0.35), ("IH", 0.20)],
    "AE": [("EH", 0.35), ("AH", 0.20)],
    # Low / back vowels.
    "AA": [("AH", 0.30), ("AO", 0.30)],
    "AH": [("AA", 0.25), ("EH", 0.15), ("ER", 0.10)],
    "AO": [("OW", 0.25), ("AA", 0.25)],
    "OW": [("AO", 0.30), ("UW", 0.20)],
    "UW": [("UH", 0.35), ("OW", 0.20)],
    "UH": [("UW", 0.35), ("AH", 0.15)],
    # Diphthongs.
    "AY": [("AH", 0.20), ("EY", 0.15)],
    "AW": [("AA", 0.20), ("OW", 0.15)],
    "OY": [("OW", 0.25), ("AY", 0.15)],
    # Rhotics.
    "ER": [("AH", 0.25), ("R", 0.20)],
}


def _normalise_confusion_map(
    raw: Dict[str, Iterable[Tuple[str, float]]],
) -> Dict[str, List[Tuple[str, float]]]:
    """Normalise weights so they sum to 1.0 per source phoneme.

    Accepts any iterable of ``(cand, weight)`` tuples. Non-positive
    total weight raises ``ValueError``.
    """
    normalised: Dict[str, List[Tuple[str, float]]] = {}
    for src, pairs in raw.items():
        pairs = list(pairs)
        total = sum(w for _, w in pairs)
        if total <= 0:
            raise ValueError(
                f"Confusion weights for phoneme {src!r} must be positive, got {pairs!r}"
            )
        normalised[src] = [(c, w / total) for c, w in pairs]
    return normalised


# ---------------------------------------------------------------------------
# Augmenter
# ---------------------------------------------------------------------------


@dataclass
class PhonemeNoiseAugmenter:
    """Apply visual-phoneme-aware noise to a phoneme sequence.

    Parameters
    ----------
    substitute_prob:
        Per-phoneme probability of a substitution. Candidates are drawn
        from ``confusion_map`` if the source phoneme is present, otherwise
        from ``fallback_vocab`` uniformly.
    insert_prob:
        Per-phoneme probability of inserting an extra phoneme before the
        current one. The inserted phoneme is either sampled from the
        confusion map (visual neighbour of the current phoneme) or
        uniformly from ``fallback_vocab``.
    delete_prob:
        Per-phoneme probability of deleting the current phoneme entirely.
    confusion_map:
        A mapping ``phoneme -> list[(candidate, weight)]``. If ``None``,
        :data:`DEFAULT_VISUAL_CONFUSION_MAP` is used.
    fallback_vocab:
        The universe of phonemes used when a source phoneme has no entry
        in ``confusion_map`` (substitution) or for "random" insertions.
        If ``None``, the union of all keys and candidates in the
        confusion map is used.
    seed:
        Optional seed for deterministic augmentation. If ``None`` a
        non-deterministic ``random.Random`` is used.

    Notes
    -----
    Order of operations per position: delete, then substitute, then
    insert (before the current phoneme). This matches the way CTC
    decoders tend to emit spurious insertions around confusable
    neighbours.
    """

    substitute_prob: float = 0.15
    insert_prob: float = 0.05
    delete_prob: float = 0.05
    confusion_map: Optional[Dict[str, Iterable[Tuple[str, float]]]] = None
    fallback_vocab: Optional[Sequence[str]] = None
    seed: Optional[int] = None

    # Initialised in __post_init__.
    _rng: random.Random = field(init=False, repr=False)
    _map: Dict[str, List[Tuple[str, float]]] = field(init=False, repr=False)
    _fallback: List[str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        for name, p in (
            ("substitute_prob", self.substitute_prob),
            ("insert_prob", self.insert_prob),
            ("delete_prob", self.delete_prob),
        ):
            if not (0.0 <= p <= 1.0):
                raise ValueError(f"{name} must be in [0,1], got {p}")

        raw_map = self.confusion_map if self.confusion_map is not None else DEFAULT_VISUAL_CONFUSION_MAP
        self._map = _normalise_confusion_map(raw_map)

        if self.fallback_vocab is None:
            vocab = set(self._map.keys())
            for cands in self._map.values():
                vocab.update(c for c, _ in cands)
            self._fallback = sorted(vocab)
        else:
            self._fallback = list(self.fallback_vocab)

        self._rng = random.Random(self.seed) if self.seed is not None else random.Random()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reseed(self, seed: int) -> None:
        """Reseed the internal RNG (useful for reproducible epochs)."""
        self._rng = random.Random(seed)

    def with_overrides(self, **kwargs) -> "PhonemeNoiseAugmenter":
        """Return a shallow copy with the given fields overridden."""
        new = copy.copy(self)
        for k, v in kwargs.items():
            setattr(new, k, v)
        new.__post_init__()
        return new

    def __call__(self, phonemes: Sequence[str]) -> List[str]:
        return self.augment(phonemes)

    def augment(self, phonemes: Sequence[str]) -> List[str]:
        """Apply noise to a phoneme sequence and return a new list."""
        out: List[str] = []
        for ph in phonemes:
            # 1) Deletion.
            if self.delete_prob > 0 and self._rng.random() < self.delete_prob:
                continue
            # 2) Insertion (before the current phoneme).
            if self.insert_prob > 0 and self._rng.random() < self.insert_prob:
                out.append(self._sample_insertion(ph))
            # 3) Substitution.
            if self.substitute_prob > 0 and self._rng.random() < self.substitute_prob:
                out.append(self._sample_substitution(ph))
            else:
                out.append(ph)
        return out

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample_substitution(self, src: str) -> str:
        cands = self._map.get(src)
        if cands:
            return self._weighted_choice(cands)
        # Unknown source → sample uniformly from the fallback vocab, excluding src.
        pool = [p for p in self._fallback if p != src] or self._fallback
        return self._rng.choice(pool)

    def _sample_insertion(self, neighbour: str) -> str:
        cands = self._map.get(neighbour)
        if cands and self._rng.random() < 0.5:
            return self._weighted_choice(cands)
        return self._rng.choice(self._fallback)

    def _weighted_choice(self, pairs: Sequence[Tuple[str, float]]) -> str:
        r = self._rng.random()
        cum = 0.0
        for cand, w in pairs:
            cum += w
            if r <= cum:
                return cand
        return pairs[-1][0]


# ---------------------------------------------------------------------------
# Convenience constructor (used from Models/Llama.py and configs/*).
# ---------------------------------------------------------------------------


def build_augmenter_from_config(cfg: Optional[dict]) -> Optional[PhonemeNoiseAugmenter]:
    """Build an augmenter from a plain config dict, or return ``None``.

    Expected keys (all optional):
        enabled (bool), substitute_prob, insert_prob, delete_prob,
        seed, confusion_map (dict), fallback_vocab (list).

    Returns ``None`` when ``cfg`` is falsy or ``enabled`` is ``False``,
    so callers can unconditionally call this and get a backward-
    compatible (clean) training path.
    """
    if not cfg or not cfg.get("enabled", False):
        return None
    return PhonemeNoiseAugmenter(
        substitute_prob=float(cfg.get("substitute_prob", 0.15)),
        insert_prob=float(cfg.get("insert_prob", 0.05)),
        delete_prob=float(cfg.get("delete_prob", 0.05)),
        confusion_map=cfg.get("confusion_map"),
        fallback_vocab=cfg.get("fallback_vocab"),
        seed=cfg.get("seed"),
    )
