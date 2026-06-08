"""G2P (espeak) + phone-set reconciliation against the model vocab.

The model emits espeak/IPA tokens; phonemizer must produce the same symbols so GOP can
index the right posterior column. We try an exact match first, then a normalized match
(stress/length/diacritics stripped). Anything unmapped is an error, not a low score.
"""
from __future__ import annotations

import unicodedata

from phonemizer import phonemize
from phonemizer.separator import Separator

# Suprasegmentals / diacritics stripped for the normalized fallback match.
_STRIP = {"ˈ", "ˌ", "ː", "ˑ", "ʰ", "ʷ", "ʲ", "ˠ", "ˤ", "ʼ"}


def g2p(word: str) -> list[str]:
    out = phonemize(
        word.strip().lower(),
        language="en-us",
        backend="espeak",
        separator=Separator(phone=" ", word="|"),
        strip=True,
        with_stress=False,
    )
    return [p for p in out.replace("|", " ").split() if p]


def normalize(phone: str) -> str:
    decomposed = unicodedata.normalize("NFD", phone)
    return "".join(
        c for c in decomposed if c not in _STRIP and not unicodedata.combining(c)
    )


def build_maps(vocab: dict[str, int]) -> tuple[dict[str, int], dict[str, int]]:
    """Build (exact, normalized) phone -> id lookups from the model vocab."""
    exact = dict(vocab)
    norm: dict[str, int] = {}
    for phone, idx in vocab.items():
        norm.setdefault(normalize(phone), idx)
    return exact, norm


def to_token_ids(
    phones: list[str], exact: dict[str, int], norm: dict[str, int]
) -> tuple[list[int], list[str]]:
    ids: list[int] = []
    unmapped: list[str] = []
    for p in phones:
        if p in exact:
            ids.append(exact[p])
        elif normalize(p) in norm:
            ids.append(norm[normalize(p)])
        else:
            unmapped.append(p)
    return ids, unmapped
