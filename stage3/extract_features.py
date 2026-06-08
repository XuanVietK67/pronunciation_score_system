"""Stage 3, step 2: extract per-phone GOP features + labels from speechocean762.

Per utterance: decode audio (soundfile) -> Stage-1 log-posteriors -> bridge ARPAbet phones
to model tokens -> forced-align -> cache per-phone LPP feature vectors with float labels.

Usage (from repo root):
  python -m stage3.extract_features --out features_cache [--limit N] [--splits train test]

Caching means we pay the (slow, CPU) forward passes once, then iterate on the model cheaply.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from datasets import Audio, load_dataset

from app import audio as audio_io
from app.acoustic import Acoustic
from app.g2p import build_maps
from app.gop import align_spans
from stage3.arpabet_ipa import arpabet_to_token_ids, strip_stress


def _audio_bytes(cell: dict) -> bytes:
    data = cell.get("bytes")
    if data is None:
        with open(cell["path"], "rb") as f:
            data = f.read()
    return data


def _bridge(words, exact, norm, dropped: Counter) -> tuple[list[int], list[float]]:
    """ARPAbet phones + per-phone labels -> model token ids + expanded labels."""
    token_ids: list[int] = []
    labels: list[float] = []
    for w in words:
        phones = w.get("phones") or []
        accs = w.get("phones-accuracy") or []
        for phone, acc in zip(phones, accs):
            ids = arpabet_to_token_ids(phone, exact, norm)
            if not ids:
                dropped[strip_stress(phone)] += 1
                continue
            for tid in ids:  # diphthong/affricate: each token inherits the phone's label
                token_ids.append(tid)
                labels.append(float(acc))
    return token_ids, labels


def process_split(ds_split, split, acoustic, exact, norm, out_dir: Path, limit) -> Counter:
    split_dir = out_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    stats: Counter = Counter()
    dropped: Counter = Counter()

    n = len(ds_split) if limit is None else min(limit, len(ds_split))
    for i in range(n):
        ex = ds_split[i]
        token_ids, labels = _bridge(ex["words"], exact, norm, dropped)
        if not token_ids:
            stats["skip_no_phones"] += 1
            continue
        try:
            wav = audio_io.load_16k_mono(_audio_bytes(ex["audio"]))
        except Exception:
            stats["skip_audio_error"] += 1
            continue

        log_probs = acoustic.log_posteriors(wav)
        if len(token_ids) > log_probs.shape[0]:
            stats["skip_too_short"] += 1
            continue

        spans = align_spans(log_probs, token_ids, acoustic.blank_id)
        if [s[0] for s in spans] != token_ids:
            # merge_tokens collapsed adjacent identical tokens -> labels would misalign
            stats["skip_merge_mismatch"] += 1
            continue

        features = torch.stack([s[3] for s in spans]).to(torch.float32)  # [L, V]
        record = {
            "features": features,
            "phone_ids": torch.tensor(token_ids, dtype=torch.long),  # [L]
            "labels": torch.tensor(labels, dtype=torch.float32),      # [L]
        }
        uid = f"{split}_{i:05d}"
        path = split_dir / f"{uid}.pt"
        torch.save(record, path)
        manifest.append({"id": uid, "path": str(path), "n_phones": len(token_ids)})
        stats["ok"] += 1
        if stats["ok"] % 200 == 0:
            print(f"  [{split}] {stats['ok']} ok / {i + 1} seen")

    (out_dir / f"{split}_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[{split}] done: {dict(stats)}")
    if dropped:
        print(f"[{split}] dropped phones: {dict(dropped)}")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="features_cache")
    ap.add_argument("--limit", type=int, default=None, help="cap utterances per split (smoke test)")
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    args = ap.parse_args()

    out_dir = Path(args.out)
    print("Loading speechocean762...")
    try:
        ds = load_dataset("mispeech/speechocean762")
    except Exception:
        ds = load_dataset("mispeech/speechocean762", trust_remote_code=True)
    ds = ds.cast_column("audio", Audio(decode=False))

    print("Loading Stage-1 model...")
    acoustic = Acoustic()
    exact, norm = build_maps(acoustic.vocab)

    for split in args.splits:
        process_split(ds[split], split, acoustic, exact, norm, out_dir, args.limit)


if __name__ == "__main__":
    main()
