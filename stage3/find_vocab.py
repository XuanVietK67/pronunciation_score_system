"""Print model vocab tokens containing any of the given characters.

Used to close phone-bridge gaps: e.g. find which rhotic/central-vowel symbol the model
actually uses for ARPAbet ER.

Usage (from repo root):  python -m stage3.find_vocab [chars]
  default chars: ɝɜɚəɹrɐ
"""
from __future__ import annotations

import sys

from app.acoustic import Acoustic


def main(chars: str) -> None:
    acoustic = Acoustic()
    hits = sorted(tok for tok in acoustic.vocab if any(c in tok for c in chars))
    print(f"vocab size: {len(acoustic.vocab)}  matches for [{chars}]: {len(hits)}")
    for tok in hits:
        print(f"  {tok!r}  id={acoustic.vocab[tok]}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "ɝɜɚəɹrɐ")
