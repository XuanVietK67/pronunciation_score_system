"""Export inference artifacts for the trained GOPT head — WITHOUT retraining.

``compute_feature_stats`` is a deterministic full pass over the cached feature manifest,
so the train ``mean``/``std`` for an already-trained checkpoint can be regenerated
exactly. This selects the best-PCC checkpoint under ``save_folder``, and writes a flat,
service-ready bundle the Dockerfile ships:

    artifacts/
      model.ckpt          # GOPT state_dict (best PCC)
      feature_stats.pt    # train mean/std
      model_config.json   # GOPT constructor args

Run this when the feature cache is unchanged but the checkpoint lacks artifacts (i.e.
it predates the stats-persistence change in train.py). If the cache changed, retrain.

Usage (from repo root):
  python -m stage3.export_artifacts stage3/hparams/gopt.yaml [--out artifacts]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.checkpoints import Checkpointer

from stage3.artifacts import CHECKPOINT_NAME, save_feature_stats, save_model_config
from stage3.dataset import compute_feature_stats
from stage3.model import GOPT


def main(hparams_file: str, out_dir: str) -> None:
    with open(hparams_file) as f:
        hparams = load_hyperpyyaml(f)

    print("Computing feature normalization stats over train (deterministic)...")
    mean, std = compute_feature_stats(hparams["train_manifest"])
    input_dim = mean.numel()
    print(f"  feature dim: {input_dim}")

    save_feature_stats(out_dir, mean, std)
    cfg = save_model_config(
        out_dir,
        input_dim=input_dim,
        vocab_size=hparams["vocab_size"],
        d_model=hparams["d_model"],
        nhead=hparams["nhead"],
        num_layers=hparams["num_layers"],
        dropout=hparams["dropout"],
    )

    model = GOPT(**cfg)
    ckptr = Checkpointer(hparams["save_folder"], recoverables={"model": model})
    best = ckptr.find_checkpoint(max_key="PCC")
    if best is None:
        raise SystemExit(f"no checkpoint found under {hparams['save_folder']}")
    print(f"Selected checkpoint {best.path.name} (PCC={best.meta.get('PCC')})")
    ckptr.load_checkpoint(best)  # loads best weights into `model`
    model.eval()

    out_path = Path(out_dir) / CHECKPOINT_NAME
    torch.save(model.state_dict(), out_path)
    print(f"Wrote {out_path} + feature_stats.pt + model_config.json to {out_dir}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("hparams", nargs="?", default="stage3/hparams/gopt.yaml")
    ap.add_argument("--out", default="artifacts")
    args = ap.parse_args()
    main(args.hparams, args.out)
