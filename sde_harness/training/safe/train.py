"""Train a frozen-prefix outcome-risk SAFE classifier.

This is the fast SAFE view only: it receives the last-layer Hy-VLA prefix
sequence cached by :mod:`cache_prefix`, learns a risk logit, and never updates
the VLA checkpoint.  Explicit safety-window labels are preferred by manifest
construction; recorder-only datasets train terminal task-failure risk.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


def _load_manifest(path: Path, split: str) -> list[dict[str, Any]]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("split") == split:
            records.append(record)
    return records


class PrefixDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        with h5py.File(record["prefix_hdf5"], "r") as cache:
            bits = np.asarray(cache["prefix/value"][record["prefix_index"]], dtype=np.uint16)
            mask = np.asarray(cache["prefix/pad_mask"][record["prefix_index"]], dtype=np.bool_)
        # HDF5 has no portable bfloat16 dtype, so caches retain the exact
        # bit pattern in uint16 and decode it here.
        prefix = torch.from_numpy(bits.copy()).view(torch.bfloat16).float()
        return {
            "prefix": prefix,
            "pad_mask": torch.from_numpy(mask.copy()),
            "risk": torch.tensor(float(record["risk_label"]), dtype=torch.float32),
        }


class PrefixRiskMLP(nn.Module):
    """Masked-prefix pooling followed by a small calibrated-risk head."""

    def __init__(self, hidden_size: int, width: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, 1),
        )

    def forward(self, prefix: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        weights = pad_mask.to(dtype=prefix.dtype).unsqueeze(-1)
        pooled = (prefix * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return self.head(pooled).squeeze(-1)


@torch.no_grad()
def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    total = correct = positives = true_positives = 0
    total_loss = 0.0
    loss_fn = nn.BCEWithLogitsLoss()
    for batch in loader:
        prefix = batch["prefix"].to(device)
        mask = batch["pad_mask"].to(device)
        risk = batch["risk"].to(device)
        logits = model(prefix, mask)
        total_loss += float(loss_fn(logits, risk).item()) * len(risk)
        prediction = logits.sigmoid() >= 0.5
        total += len(risk)
        correct += int((prediction == risk.bool()).sum())
        positives += int(risk.bool().sum())
        true_positives += int((prediction & risk.bool()).sum())
    if not total:
        return {"samples": 0.0}
    return {
        "samples": float(total),
        "loss": total_loss / total,
        "accuracy": correct / total,
        "risk_recall": true_positives / positives if positives else float("nan"),
    }


def train(
    *,
    manifest: Path,
    output: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    width: int,
    dropout: float,
    seed: int,
    device_name: str,
) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(device_name)

    train_records = _load_manifest(manifest, "train")
    val_records = _load_manifest(manifest, "val")
    if not train_records:
        raise ValueError("manifest has no train records")
    label_counts = Counter(int(record["risk_label"]) for record in train_records)
    if set(label_counts) != {0, 1}:
        raise ValueError(
            "SAFE binary training requires both risk labels in the train split; "
            f"got {dict(label_counts)}. Collect failed/unsafe episodes or add explicit safety labels."
        )

    train_dataset = PrefixDataset(train_records)
    val_dataset = PrefixDataset(val_records)
    first = train_dataset[0]
    hidden_size = int(first["prefix"].shape[-1])
    model = PrefixRiskMLP(hidden_size, width=width, dropout=dropout).to(device)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    positive_weight = torch.tensor(
        [label_counts[0] / label_counts[1]], dtype=torch.float32, device=device
    )
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        samples = 0
        for batch in train_loader:
            prefix = batch["prefix"].to(device)
            mask = batch["pad_mask"].to(device)
            risk = batch["risk"].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(prefix, mask), risk)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(risk)
            samples += len(risk)
        summary = {
            "epoch": epoch,
            "train_loss": total_loss / max(samples, 1),
            "val": _evaluate(model, val_loader, device),
        }
        history.append(summary)
        print(json.dumps(summary))

    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "architecture": "prefix_risk_mlp",
        "hidden_size": hidden_size,
        "width": width,
        "dropout": dropout,
        "manifest": str(manifest.resolve()),
        "label_counts": dict(label_counts),
        "state_dict": model.state_dict(),
    }
    torch.save(payload, output / "safe_prefix_risk.pt")
    report = {"history": history, "train_samples": len(train_records), "val_samples": len(val_records)}
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = train(
        manifest=Path(args.manifest).expanduser().resolve(),
        output=Path(args.output).expanduser().resolve(),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        width=args.width,
        dropout=args.dropout,
        seed=args.seed,
        device_name=args.device,
    )
    print(json.dumps(report))


if __name__ == "__main__":
    main()
