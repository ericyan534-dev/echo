"""Train FillerNet on the PodcastFillers clips (cut by scripts/fetch_pfsd.py).

Usage:
    python scripts/train_filler.py                 # train + eval, save checkpoint
    python scripts/train_filler.py --eval-only     # evaluate existing checkpoint
    python scripts/train_filler.py --epochs 4 --limit 8000   # quick smoke run

Gate (plan §3): binary filler (uh∪um) F1 >= 0.75 on the official test split.
Outputs: models/fillernet.pt, models/fillernet_metrics.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.acoustic.features import logmel  # noqa: E402
from backend.acoustic.model import CLASSES, LABEL_MAP, FillerNet  # noqa: E402

CLIPS = ROOT / "data" / "pfsd" / "clips"
CSV_PATH = ROOT / "data" / "pfsd" / "PodcastFillers.csv"
MODELS = ROOT / "models"
SR = 16_000


def extra_train_episode_names() -> set[str]:
    """clip_names of `extra` clips whose *episode* is in the train split.

    PFSD's `extra` subset spans all 199 episodes — including the 20 test and 6
    validation episodes (923 extra clips sit in test episodes; 255 of them
    overlap a test clip's window). Training on unfiltered `extra` therefore
    breaks episode-disjointness with the test set. Keep only extra clips from
    train-split episodes (7,917 of 9,114).
    """
    import csv

    keep: set[str] = set()
    with open(CSV_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (row["clip_split_subset"] == "extra"
                    and row["episode_split_subset"] == "train"):
                keep.add(row["clip_name"])
    return keep


def scan_split(split_dirs: list[str], limit: int = 0,
               extra_allow: set[str] | None = None) -> list[tuple[Path, int]]:
    items: list[tuple[Path, int]] = []
    for split in split_dirs:
        base = CLIPS / split
        if not base.is_dir():
            continue
        for label_dir in base.iterdir():
            cls = LABEL_MAP.get(label_dir.name)
            if cls is None:
                continue
            idx = CLASSES.index(cls)
            for p in label_dir.glob("*.wav"):
                if split == "extra" and extra_allow is not None and p.name not in extra_allow:
                    continue
                items.append((p, idx))
    random.Random(13).shuffle(items)
    return items[:limit] if limit else items


def load_features(items: list[tuple[Path, int]], desc: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode + log-mel all clips into RAM (85k clips ~= 2.2 GB float32)."""
    t0 = time.time()

    def one(pair: tuple[Path, int]) -> tuple[np.ndarray, int]:
        path, y = pair
        x, _ = sf.read(path, dtype="float32")
        if len(x) < SR:
            x = np.pad(x, (0, SR - len(x)))
        feats = logmel(torch.from_numpy(x[:SR]))
        return feats.numpy(), y

    with ThreadPoolExecutor(max_workers=12) as ex:
        results = list(ex.map(one, items))
    X = torch.from_numpy(np.stack([r[0] for r in results])).unsqueeze(1)
    y = torch.tensor([r[1] for r in results], dtype=torch.long)
    print(f"  {desc}: {len(items)} clips featurized in {time.time()-t0:.0f}s "
          f"({X.shape}, {X.element_size()*X.nelement()/1e9:.1f} GB)", flush=True)
    return X, y


def augment(xb: torch.Tensor) -> torch.Tensor:
    """Cheap spec augment: time roll + gain jitter + light freq/time masking."""
    xb = torch.roll(xb, shifts=int(torch.randint(-10, 11, (1,))), dims=-1)
    xb = xb * (1.0 + 0.1 * torch.randn(xb.size(0), 1, 1, 1, device=xb.device))
    if random.random() < 0.5:  # frequency mask
        f0 = random.randint(0, 56)
        xb[:, :, f0:f0 + 8, :] = 0
    if random.random() < 0.5:  # time mask
        t0 = random.randint(0, 89)
        xb[:, :, :, t0:t0 + 12] = 0
    return xb


@torch.no_grad()
def evaluate(model: nn.Module, X: torch.Tensor, y: torch.Tensor, device: str) -> dict:
    model.eval()
    preds = []
    for i in range(0, len(X), 512):
        preds.append(model(X[i:i + 512].to(device)).argmax(1).cpu())
    p = torch.cat(preds)

    def f1(binary_true: torch.Tensor, binary_pred: torch.Tensor) -> tuple[float, float, float]:
        tp = int((binary_pred & binary_true).sum())
        fp = int((binary_pred & ~binary_true).sum())
        fn = int((~binary_pred & binary_true).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        return (2 * prec * rec / (prec + rec) if prec + rec else 0.0, prec, rec)

    out: dict = {"accuracy": float((p == y).float().mean()), "per_class": {}}
    for i, c in enumerate(CLASSES):
        s, prec, rec = f1(y == i, p == i)
        out["per_class"][c] = {"f1": round(s, 4), "precision": round(prec, 4),
                               "recall": round(rec, 4), "n": int((y == i).sum())}
    uh, um = CLASSES.index("uh"), CLASSES.index("um")
    s, prec, rec = f1((y == uh) | (y == um), (p == uh) | (p == um))
    out["filler_binary"] = {"f1": round(s, 4), "precision": round(prec, 4), "recall": round(rec, 4)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--limit", type=int, default=0, help="cap train clips (smoke run)")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--aux-binary", type=float, default=0.0,
                    help="weight of an auxiliary binary (uh|um vs speech/other) BCE loss")
    ap.add_argument("--uh-boost", type=float, default=1.0,
                    help="multiply the uh class weight by this factor before normalization")
    ap.add_argument("--out", type=str, default=str(MODELS / "fillernet.pt"),
                    help="checkpoint output path (metrics json is written as <stem>_metrics.json)")
    ap.add_argument("--no-test", action="store_true",
                    help="skip the final TEST metrics eval + gate; report/save VAL metrics only "
                         "(select-on-validation, for grid runs)")
    args = ap.parse_args()

    print(f"flags: aux-binary={args.aux_binary} uh-boost={args.uh_boost} "
          f"out={args.out} no-test={args.no_test}", flush=True)

    random.seed(13)
    np.random.seed(13)
    torch.manual_seed(13)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)
    ckpt_path = Path(args.out)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = ckpt_path.parent / f"{ckpt_path.stem}_metrics.json"

    need_test = args.eval_only or not args.no_test
    Xte = yte = None
    if need_test:
        test_items = scan_split(["test"])
        if not test_items:
            print("FATAL: no test clips found -- wait for scripts/fetch_pfsd.py to finish.")
            return 2
        Xte, yte = load_features(test_items, "test")

    if args.eval_only:
        from backend.acoustic.model import load_checkpoint

        model = load_checkpoint(ckpt_path, device)
        metrics = evaluate(model, Xte, yte, device)
        print(json.dumps(metrics, indent=2))
        gate = metrics["filler_binary"]["f1"]
        print(f"\nGATE filler-F1 >= 0.75: {'PASS' if gate >= 0.75 else 'FAIL'} ({gate:.3f})")
        return 0 if gate >= 0.75 else 1

    train_items = scan_split(["train", "extra"], limit=args.limit,
                             extra_allow=extra_train_episode_names())
    val_items = scan_split(["validation"])
    Xtr, ytr = load_features(train_items, "train")
    Xva, yva = load_features(val_items, "validation")

    model = FillerNet().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"FillerNet params: {n_params:,}", flush=True)

    # class weights (inverse frequency, normalized)
    counts = torch.bincount(ytr, minlength=len(CLASSES)).float()
    weights = (counts.sum() / counts.clamp_min(1)).to(device)
    uh_idx, um_idx = CLASSES.index("uh"), CLASSES.index("um")
    speech_idx, other_idx = CLASSES.index("speech"), CLASSES.index("other")
    weights[uh_idx] *= args.uh_boost
    weights = weights / weights.mean()
    crit = nn.CrossEntropyLoss(weight=weights)
    # The product trigger fires on uh-union-um, so the binary margin is the
    # deployed metric; uh/um confusion is free -- optionally shape the loss to match.
    aux_crit = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = max(1, -(-len(Xtr) // args.batch))  # ceil — must match the loop's batch count
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr,
                                                total_steps=args.epochs * steps)

    best_f1, best_state = 0.0, None
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        total_loss = 0.0
        for i in range(0, len(perm), args.batch):
            idx = perm[i:i + args.batch]
            xb = augment(Xtr[idx].to(device, non_blocking=True))
            yb = ytr[idx].to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = crit(logits, yb)
            if args.aux_binary:
                binary_logit = (torch.logsumexp(logits[:, [uh_idx, um_idx]], dim=1)
                                - torch.logsumexp(logits[:, [speech_idx, other_idx]], dim=1))
                is_filler = ((yb == uh_idx) | (yb == um_idx)).float()
                loss = loss + args.aux_binary * aux_crit(binary_logit, is_filler)
            loss.backward()
            opt.step()
            sched.step()
            total_loss += float(loss) * len(idx)
        val = evaluate(model, Xva, yva, device)
        vf1 = val["filler_binary"]["f1"]
        print(f"epoch {epoch+1:2d}/{args.epochs}  loss {total_loss/len(Xtr):.4f}  "
              f"val filler-F1 {vf1:.4f}  val acc {val['accuracy']:.4f}", flush=True)
        if vf1 > best_f1:
            best_f1 = vf1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)

    if args.no_test:
        val_metrics = evaluate(model, Xva, yva, device)
        print("\nVAL metrics:")
        print(json.dumps(val_metrics, indent=2))
        torch.save({"model": model.state_dict(), "classes": CLASSES,
                    "val_metrics": val_metrics, "params": n_params}, ckpt_path)
        metrics_path.write_text(json.dumps(val_metrics, indent=2), encoding="utf-8")
        print(f"\nsaved {ckpt_path}")
        return 0

    metrics = evaluate(model, Xte, yte, device)
    print("\nTEST metrics:")
    print(json.dumps(metrics, indent=2))

    torch.save({"model": model.state_dict(), "classes": CLASSES,
                "metrics": metrics, "params": n_params}, ckpt_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nsaved {ckpt_path}")
    gate = metrics["filler_binary"]["f1"]
    print(f"GATE filler-F1 >= 0.75: {'PASS' if gate >= 0.75 else 'FAIL'} ({gate:.3f})")
    return 0 if gate >= 0.75 else 1


if __name__ == "__main__":
    sys.exit(main())
