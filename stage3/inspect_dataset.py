"""Stage 3, step 1: confirm the speechocean762 schema and measure phone-bridge coverage.

This is the go/no-go gate before building the GOPT head. It:
  1. prints split sizes, the feature schema, and one example's keys,
  2. tallies the per-phone label distribution (expecting {0, 1, 2}),
  3. reports what fraction of ARPAbet phones bridge to model tokens.

Usage (from repo root):  python -m stage3.inspect_dataset
"""
from __future__ import annotations

from collections import Counter

from datasets import Audio, load_dataset

from app.acoustic import Acoustic
from app.g2p import build_maps
from stage3.arpabet_ipa import arpabet_to_ipa, arpabet_to_token_ids, strip_stress


def _words(example: dict) -> list:
    return example.get("words") or example.get("word") or []


def _phones(word: dict) -> list:
    return word.get("phones") or word.get("phone") or []


def _phone_acc(word: dict) -> list:
    for key in ("phones-accuracy", "phones_accuracy", "phone_accuracy", "phones_acc"):
        if key in word:
            return word[key]
    return []


def main() -> None:
    print("Loading speechocean762 (first run downloads it)...")
    try:
        ds = load_dataset("mispeech/speechocean762")
    except Exception:  # some versions need trust_remote_code
        ds = load_dataset("mispeech/speechocean762", trust_remote_code=True)

    # Don't let `datasets` decode audio (avoids the torchcodec dependency); we decode the
    # raw bytes ourselves with soundfile when we actually need the waveform.
    ds = ds.cast_column("audio", Audio(decode=False))

    print("\n=== splits ===")
    for split in ds:
        print(f"  {split}: {len(ds[split])}")

    test = ds["test"]
    print("\n=== features ===")
    print(test.features)

    example = test[0]
    print("\n=== example[0] keys ===")
    print(list(example.keys()))
    first_words = _words(example)
    if first_words:
        print("first word keys:", list(first_words[0].keys()))
        print("first word:", first_words[0])

    # --- label distribution + ARPAbet inventory (no audio decode: read the column only) ---
    words_col = test["words"]
    labels = Counter()
    arpa_inventory = Counter()
    for words in words_col:
        for w in words:
            for p, a in zip(_phones(w), _phone_acc(w)):
                arpa_inventory[strip_stress(p)] += 1
                labels[round(float(a))] += 1

    print("\n=== phone-label distribution (test) ===")
    for k in sorted(labels):
        print(f"  {k}: {labels[k]}")

    # --- bridge coverage ---
    print("\nLoading Stage-1 model to read its vocab...")
    acoustic = Acoustic()
    exact, norm = build_maps(acoustic.vocab)

    mapped, unmapped = [], []
    for arpa in sorted(arpa_inventory):
        ids = arpabet_to_token_ids(arpa, exact, norm)
        (mapped if ids else unmapped).append(arpa)

    covered = sum(arpa_inventory[a] for a in mapped)
    total = sum(arpa_inventory.values())
    print("\n=== phone-bridge coverage ===")
    print(f"  ARPAbet phone types: {len(arpa_inventory)}  mapped: {len(mapped)}")
    print(f"  phone tokens covered: {covered}/{total}  ({100 * covered / max(total, 1):.1f}%)")
    if unmapped:
        print("  UNMAPPED phone types (label, IPA):")
        for a in unmapped:
            print(f"    {a} -> {arpabet_to_ipa(a)}  (count {arpa_inventory[a]})")


if __name__ == "__main__":
    main()
