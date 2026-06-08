"""Step-1 smoke check: print the collapsed phone sequence the model hears for a wav.

Usage (from repo root):  python -m scripts.inspect_acoustic path/to/audio.wav
"""
from __future__ import annotations

import sys

from app import audio
from app.acoustic import Acoustic


def main(path: str) -> None:
    acoustic = Acoustic()
    with open(path, "rb") as f:
        wav = audio.load_16k_mono(f.read())

    log_probs = acoustic.log_posteriors(wav)
    top = log_probs.argmax(-1).tolist()

    # CTC-collapse: drop blanks and consecutive repeats for a readable view.
    seq, prev = [], None
    for idx in top:
        if idx != prev and idx != acoustic.blank_id:
            seq.append(acoustic.id2phone[idx])
        prev = idx

    print(f"frames: {log_probs.shape[0]}  vocab: {log_probs.shape[1]}")
    print("heard:", " ".join(seq))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m scripts.inspect_acoustic path/to/audio.wav")
    main(sys.argv[1])
