"""Run the full scoring pipeline on a local audio file and print JSON.

Usage (from repo root):  python -m scripts.score_file path/to/audio.wav "word"

Same pipeline as POST /score, minus HTTP — handy for quick iteration. Honors
USE_TRAINED_HEAD (loads the trained GOPT head unless disabled).
"""
from __future__ import annotations

import json
import sys

from app import audio, config
from app.acoustic import Acoustic
from app.g2p import build_maps
from app.score_pipeline import score_word
from app.scorer import load_scorer


def main(path: str, word: str) -> None:
    acoustic = Acoustic()
    exact, norm = build_maps(acoustic.vocab)
    scorer = load_scorer() if config.USE_TRAINED_HEAD else None

    with open(path, "rb") as f:
        wav = audio.load_16k_mono(f.read())

    phones, phonemes, overall = score_word(wav, word, acoustic, exact, norm, scorer)

    print(
        json.dumps(
            {
                "word": word,
                "transcript_phonemes": phones,
                "overall_score": overall,
                "phonemes": phonemes,
                "model_version": config.MODEL_VERSION,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit('usage: python -m scripts.score_file path/to/audio.wav "word"')
    main(sys.argv[1], sys.argv[2])
