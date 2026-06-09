"""Stage 3 training — GOPT head in SpeechBrain's Brain loop.

Usage (from repo root):  python -m stage3.train stage3/hparams/gopt.yaml

speechocean762 ships only train/test, so we use the test split as the validation/eval set
and report phoneme-level PCC on it (standard for this benchmark).
"""
from __future__ import annotations

import sys

import speechbrain as sb
import torch
import torch.nn.functional as F
from hyperpyyaml import load_hyperpyyaml
from scipy.stats import pearsonr
from torch.utils.data import DataLoader

from stage3.artifacts import save_feature_stats, save_model_config
from stage3.dataset import GOPDataset, collate, compute_feature_stats
from stage3.model import GOPT


class GOPTBrain(sb.Brain):
    def compute_forward(self, batch, stage):
        features, phone_ids, labels, pad_mask, _ = batch
        features = features.to(self.device)
        phone_ids = phone_ids.to(self.device)
        pad_mask = pad_mask.to(self.device)
        preds = self.modules.model(features, phone_ids, padding_mask=pad_mask)
        return preds, pad_mask, labels.to(self.device)

    def compute_objectives(self, predictions, batch, stage):
        preds, pad_mask, labels = predictions
        valid = ~pad_mask
        p, y = preds[valid], labels[valid]
        # Upweight rare low-accuracy phones so the model doesn't collapse to the majority (2.0).
        w = 1.0 + self.hparams.imbalance_k * (2.0 - y)
        loss = (w * (p - y) ** 2).sum() / w.sum()
        if stage != sb.Stage.TRAIN:
            self.val_preds.extend(p.detach().cpu().tolist())
            self.val_labels.extend(y.detach().cpu().tolist())
        return loss

    def on_stage_start(self, stage, epoch=None):
        self.val_preds, self.val_labels = [], []

    def on_stage_end(self, stage, stage_loss, epoch=None):
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
            return
        pcc = pearsonr(self.val_preds, self.val_labels)[0] if len(self.val_preds) > 1 else 0.0
        print(f"epoch {epoch} | {stage.name} | loss {stage_loss:.4f} | PCC {pcc:.4f}")
        if stage == sb.Stage.VALID and self.checkpointer is not None:
            self.checkpointer.save_and_keep_only(meta={"PCC": pcc}, max_keys=["PCC"])


def main(hparams_file: str) -> None:
    with open(hparams_file) as f:
        hparams = load_hyperpyyaml(f)

    print("Computing feature normalization stats over train...")
    mean, std = compute_feature_stats(hparams["train_manifest"])
    input_dim = mean.numel()
    print(f"  feature dim: {input_dim}")

    # Persist exactly what the service needs to mirror this run at inference time.
    save_feature_stats(hparams["output_folder"], mean, std)
    save_model_config(
        hparams["output_folder"],
        input_dim=input_dim,
        vocab_size=hparams["vocab_size"],
        d_model=hparams["d_model"],
        nhead=hparams["nhead"],
        num_layers=hparams["num_layers"],
        dropout=hparams["dropout"],
    )

    model = GOPT(
        input_dim=input_dim,
        vocab_size=hparams["vocab_size"],
        d_model=hparams["d_model"],
        nhead=hparams["nhead"],
        num_layers=hparams["num_layers"],
        dropout=hparams["dropout"],
    )
    hparams["checkpointer"].add_recoverable("model", model)

    train_loader = DataLoader(
        GOPDataset(hparams["train_manifest"], mean, std),
        batch_size=hparams["batch_size"],
        shuffle=True,
        collate_fn=collate,
    )
    valid_loader = DataLoader(
        GOPDataset(hparams["test_manifest"], mean, std),
        batch_size=hparams["batch_size"],
        shuffle=False,
        collate_fn=collate,
    )

    print(f"  train utts: {len(train_loader.dataset)} | valid utts: {len(valid_loader.dataset)}")

    brain = GOPTBrain(
        modules={"model": model},
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts={"device": hparams.get("device", "cpu")},
        checkpointer=hparams["checkpointer"],
    )
    brain.fit(hparams["epoch_counter"], train_loader, valid_loader)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m stage3.train stage3/hparams/gopt.yaml")
    main(sys.argv[1])
