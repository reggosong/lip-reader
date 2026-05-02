"""
Lightweight phoneme utilities shared across training, evaluation and
debug scripts.

These helpers are deliberately split out of :mod:`Models.Llama` (which
imports heavy deps like ``transformers``, ``datasets`` and ``peft``)
so callers that only need text-to-phoneme conversion – unit tests,
debug scripts, the augmenter, etc. – can pull them in cheaply.
"""

from __future__ import annotations

import re
from typing import List

import pronouncing


_WORD_RE = re.compile(r"[A-Za-z']+")


def strip_stress(arpabet: str) -> str:
    """Remove CMU stress digits (0/1/2) from an ARPAbet phoneme."""
    return re.sub(r"\d", "", arpabet)


def text_to_arpabet_words(text: str) -> List[str]:
    """Convert a text string into a list of per-word ARPAbet strings.

    Out-of-vocabulary words fall back to a ``UNK(...)`` marker so the
    caller can detect and handle them without losing alignment.
    """
    words = _WORD_RE.findall(text)
    out: List[str] = []
    for w in words:
        lw = w.lower()
        phones = pronouncing.phones_for_word(lw)
        if phones:
            out.append(strip_stress(phones[0]))
        else:
            out.append(f"UNK({lw})")
    return out


def text_to_phoneme_line(text: str) -> str:
    """Format phonemes for a sentence as a space-separated string."""
    return " ".join(text_to_arpabet_words(text))


def phoneme_line_to_tokens(phon_line: str) -> List[str]:
    """Flatten a phoneme line into a list of tokens, dropping '|' separators."""
    return [t for t in phon_line.split() if t and t != "|"]
