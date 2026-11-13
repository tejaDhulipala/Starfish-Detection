"""Train the candidate-rescoring classifier with out-of-fold (OOF) scoring, then measure the F2 gain.

    python -m src.rescore.train_cls --cand ens_c04_c06_c07_c08 --imgsz 224

Model: YOLO26s-cls (ImageNet-pretrained, pretrained/yolo26s-cls.pt) with its 1000-way linear layer
replaced by 7 outputs, trained with BCE on the cumulative IoU bins [iou > 0.3, > 0.4, ..., > 0.9]
(src.rescore.make_candidates). Classifier score = mean of the 7 sigmoids. Head dropout --dropout
(0.7 as in the 1st-place-style recipe; YOLO has no drop-path, so that part is skipped).

Augmentation (train only): rotation in [-45, 45] degrees on the stored 1.5x-margin crop, then centre
crop of the 1.2 * max(w, h) view, random hflip / vflip / transpose, resize to imgsz, cutout.
Pixels: RGB, /255, no mean/std (as Ultralytics classification).

For each fold k: train on the other folds' crops, score fold k. The OOF scores of all folds are then
combined with the detector (fused) confidence and scored with the usual F2 cutoff sweep, against the
detector-only baseline on the same boxes. Writes model_dev/rescore/<tag>/: args.json, oof.csv,
results.json, fold<k>.pt.
"""

import argparse
import csv
import json
import math
import random
import time

import cv2
import numpy as np
import torch
import torch.nn as nn

from src.paths import DATA, PRETRAINED, ROOT

BINS = np.arange(0.3, 0.95, 0.1)  # 0.3 ... 0.9, 7 bins
MARGIN = 1.5


def load_candidates(name):
    rows = list(csv.DictReader(open(DATA / "cls" / name / "candidates.csv")))
    imgs = [cv2.imread(str(DATA / "cls" / name / "crops" / f"{r['idx']}.png"))[..., ::-1].copy() for r in rows]  # RGB
    return rows, imgs


class CropSet(torch.utils.data.Dataset):
    def __init__(self, imgs, ious, imgsz, train):
        self.imgs, self.ious, self.imgsz, self.train = imgs, ious, imgsz, train

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        img = self.imgs[i]
        s = img.shape[0]
        if self.train:
            a = random.uniform(-45, 45)
            m = cv2.getRotationMatrix2D((s / 2, s / 2), a, 1.0)
            img = cv2.warpAffine(img, m, (s, s), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))
        v = max(int(round(s / MARGIN)), 2)
        o = (s - v) // 2
        img = img[o : o + v, o : o + v]
        if self.train:
            if random.random() < 0.5:
                img = img[:, ::-1]
            if random.random() < 0.5:
                img = img[::-1]
            if random.random() < 0.5:
                img = img.transpose(1, 0, 2)
        img = cv2.resize(np.ascontiguousarray(img), (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
        if self.train and random.random() < 0.5:  # cutout: one grey square, up to 1/3 of the side
            c = random.randint(self.imgsz // 8, self.imgsz // 3)
            y, x = random.randint(0, self.imgsz - c), random.randint(0, self.imgsz - c)
            img = img.copy()
            img[y : y + c, x : x + c] = 114
        x = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).float() / 255.0
        y = torch.from_numpy((self.ious[i] > BINS).astype(np.float32))
        return x, y


def build_model(dropout):
    from ultralytics import YOLO

    model = YOLO(str(PRETRAINED / "yolo26s-cls.pt")).model
    head = model.model[-1]
    head.linear = nn.Linear(head.linear.in_features, len(BINS))
    head.drop = nn.Dropout(p=dropout)
    for p in model.parameters():
        p.requires_grad_(True)
    return model


def logits_of(model, x):
    out = model(x)
    return out if torch.is_tensor(out) else out[1]  # eval mode returns (softmax, logits)


@torch.no_grad()
def predict(model, ds, batch, dev):
    model.eval()
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=False, num_workers=4)
    out = []
    for x, _ in dl:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(torch.sigmoid(logits_of(model, x.to(dev)).float()).cpu())
    return torch.cat(out).numpy()


def train_fold(tr_imgs, tr_ious, args, dev):
    model = build_model(args.dropout).to(dev)
    ds = CropSet(tr_imgs, tr_ious, args.imgsz, train=True)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=6, drop_last=True, persistent_workers=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps, warm = args.epochs * len(dl), len(dl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: min(1.0, (t + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(t, steps) / steps)))
    loss_fn = nn.BCEWithLogitsLoss()
    for ep in range(args.epochs):
        model.train()
        tot, n = 0.0, 0
        for x, y in dl:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = loss_fn(logits_of(model, x).float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot, n = tot + loss.item() * len(x), n + len(x)
        print(f"    epoch {ep + 1}/{args.epochs} loss {tot / n:.4f}", flush=True)
    return model


def auc(pos, s):
    """ROC AUC via the rank-sum (Mann-Whitney) formula, ties averaged."""
    r = np.empty(len(s))
    order = np.argsort(s, kind="stable")
    r[order] = np.arange(1, len(s) + 1)
    for v in np.unique(s):  # average ranks of ties
        t = s == v
        if t.sum() > 1:
            r[t] = r[t].mean()
    n1, n0 = pos.sum(), (~pos).sum()
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def evaluate(rows, det, cls):
    """F2 of the detector-only and rescored candidate sets (non-candidate boxes don't exist: all fused
    boxes above make_candidates' --min-conf are candidates)."""
    from src.eval.evaluate import val_frames
    from src.eval.f2 import f2_sweep

    frames = val_frames("seq20val_pos")
    grid = np.round(np.arange(0.01, 1.0, 0.01), 2)
    box = np.array([[float(r[k]) for k in ("x", "y", "w", "h")] for r in rows])
    ids = np.array([r["image_id"] for r in rows])
    by_id = {}
    for j, i in enumerate(ids):
        by_id.setdefault(i, []).append(j)

    def score(s):
        data = []
        for iid, _, g in frames:
            j = by_id.get(iid, [])
            p = np.column_stack([s[j], box[j]]) if j else np.zeros((0, 5))
            data.append((p[np.argsort(-p[:, 0])] if len(p) else p, g))
        sw = f2_sweep(data, grid)
        b = int(np.argmax(sw["f2"]))
        return {"f2": round(float(sw["f2"][b]), 4), "conf": float(grid[b]), "tp50": int(sw["tp"][4, b]), "fp50": int(sw["fp"][4, b]), "fn50": int(sw["fn"][4, b])}

    out = {"detector": score(det), "classifier": score(cls), "mean": score((det + cls) / 2), "geomean": score(np.sqrt(det * cls))}
    for a in (0.25, 0.75):
        out[f"det^{a}*cls^{1 - a}"] = score(det**a * cls ** (1 - a))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cand", default="ens_c04_c06_c07_c08")
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--dropout", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--extra", default=None, help="extra always-train crops (e.g. train_jitter from make_train_crops)")
    ap.add_argument("--tag", default=None, help="default: <cand>_cls26s_<imgsz>[_<extra>]")
    args = ap.parse_args()

    tag = args.tag or f"{args.cand}_cls26s_{args.imgsz}" + (f"_{args.extra}" if args.extra else "")
    out = ROOT / "model_dev" / "rescore" / tag
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dev = torch.device("cuda")

    rows, imgs = load_candidates(args.cand)
    ious = np.array([float(r["iou"]) for r in rows])
    folds = np.array([int(r["fold"]) for r in rows])
    det = np.array([float(r["conf"]) for r in rows])
    x_imgs, x_ious = [], np.zeros(0)
    if args.extra:
        x_rows, x_imgs = load_candidates(args.extra)
        x_ious = np.array([float(r["iou"]) for r in x_rows])
        print(f"extra {args.extra}: {len(x_imgs)} crops ({(x_ious > 0.3).sum()} with iou > 0.3), added to every fold's training")
    oof = np.full((len(rows), len(BINS)), np.nan)
    t0 = time.time()
    for k in sorted(set(folds)):
        tr, te = np.where(folds != k)[0], np.where(folds == k)[0]
        print(f"fold {k}: train {len(tr)} crops ({(ious[tr] > 0.3).sum()} pos) + {len(x_imgs)} extra, score {len(te)}  [{time.time() - t0:.0f}s]", flush=True)
        model = train_fold([imgs[i] for i in tr] + list(x_imgs), np.concatenate([ious[tr], x_ious]), args, dev)
        oof[te] = predict(model, CropSet([imgs[i] for i in te], ious[te], args.imgsz, train=False), args.batch, dev)
        torch.save(model.state_dict(), out / f"fold{k}.pt")
        del model
        torch.cuda.empty_cache()

    cls = oof.mean(1)
    with open(out / "oof.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "image_id", "fold", "iou", "det", "cls"] + [f"p{b:.1f}" for b in BINS])
        for r, c, p in zip(rows, cls, oof):
            w.writerow([r["idx"], r["image_id"], r["fold"], r["iou"], r["conf"], f"{c:.5f}"] + [f"{v:.4f}" for v in p])
    res = evaluate(rows, det, cls)
    pos = ious > 0.5
    res["auc_iou0.5"] = {"detector": round(auc(pos, det), 4), "classifier": round(auc(pos, cls), 4)}
    res["minutes"] = round((time.time() - t0) / 60, 1)
    (out / "results.json").write_text(json.dumps(res, indent=2) + "\n")
    for k, v in res.items():
        print(f"  {k:22s} {v}")


if __name__ == "__main__":
    main()
