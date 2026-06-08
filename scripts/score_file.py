"""Run the full Stage 1+2 scoring pipeline on a local audio file and print JSON.

Usage (from repo root):  python -m scripts.score_file path/to/audio.wav "word"

This is the same pipeline as POST /score, minus HTTP — handy for quick iteration.
"""
from __future__ import annotations

import json
import sys

from app import audio, gop
from app.acoustic import Acoustic
from app.g2p import build_maps, g2p, to_token_ids


def main(path: str, word: str) -> None:
    acoustic = Acoustic()
    exact, norm = build_maps(acoustic.vocab)

    with open(path, "rb") as f:
        wav = audio.load_16k_mono(f.read())

    phones = g2p(word)
    token_ids, unmapped = to_token_ids(phones, exact, norm)
    if unmapped:
        print(f"WARNING unmapped phones for '{word}': {unmapped}", file=sys.stderr)

    log_probs = acoustic.log_posteriors(wav)
    feats = gop.align_and_score(log_probs, token_ids, acoustic.id2phone, acoustic.blank_id)

    phonemes = []
    for feat in feats:
        score = gop.gop_to_score(feat["gop"])
        phonemes.append(
            {
                "phone": feat["phone"],
                "score": score,
                "label": gop.score_to_label(score),
                "gop": round(feat["gop"], 4),
                "start_sec": feat["start_sec"],
                "end_sec": feat["end_sec"],
            }
        )
    overall = int(round(sum(p["score"] for p in phonemes) / max(len(phonemes), 1)))

    print(
        json.dumps(
            {
                "word": word,
                "transcript_phonemes": phones,
                "overall_score": overall,
                "phonemes": phonemes,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit('usage: python -m scripts.score_file path/to/audio.wav "word"')
    main(sys.argv[1], sys.argv[2])
