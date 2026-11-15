"""Where the c04+c06+c07+c08 WBF ensemble fails on val (cutoff 0.21, hit = IoU >= 0.5).

    python -m src.eval.fail_analysis

Per ground-truth box: size sqrt(w*h), sharpness (Laplacian variance), contrast (grey std), colour
separation from a surrounding ring (Lab distance), brightness, frame-edge contact, crowding, video
and sequence. Prints miss rate per bin, a logistic regression of miss on all features (to separate
confounds such as blur vs size), and a breakdown of false positives.
"""
import csv, numpy as np, cv2
from src.eval.ensemble import fused_val_preds
from src.eval.f2 import iou_matrix
from src.paths import split_dir
CUT = 0.21
fused, frames = fused_val_preds(["c04", "c06", "c07", "c08"])
meta = {r["image_id"]: r for r in csv.DictReader(open(split_dir("seq20val_pos") / "split.csv"))}
G, F = [], []   # per-GT rows, per-FP rows
for iid, path, g in frames:
    p = fused[iid]; p = p[p[:, 0] >= CUT]; p = p[np.argsort(-p[:, 0])]
    img = None
    if len(g) or len(p):
        img = cv2.imread(str(path)); gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32); lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    free = np.ones(len(g), bool); tp = np.zeros(len(p), bool)
    if len(p) and len(g):
        ious = iou_matrix(p[:, 1:], g)
        for i in range(len(p)):
            h = np.flatnonzero(free & (ious[i] >= 0.5))
            if len(h): free[h[0]] = False; tp[i] = True
    best = iou_matrix(g, p[:, 1:]).max(1) if len(g) and len(p) else np.zeros(len(g))
    gi = iou_matrix(g, g) if len(g) else None
    for k, (x, y, w, h) in enumerate(g):
        x0, y0, x1, y1 = int(max(x, 0)), int(max(y, 0)), int(min(x + w, 1280)), int(min(y + h, 720))
        crop = gray[y0:y1, x0:x1]
        pad = int(max(w, h) * 0.5)
        X0, Y0, X1, Y1 = max(x0 - pad, 0), max(y0 - pad, 0), min(x1 + pad, 1280), min(y1 + pad, 720)
        ring = np.ones((Y1 - Y0, X1 - X0), bool); ring[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0] = False
        L_in = lab[y0:y1, x0:x1].reshape(-1, 3).mean(0); L_out = lab[Y0:Y1, X0:X1][ring].mean(0)
        cx, cy = x + w / 2, y + h / 2
        crowd = sum(1 for j, (a, b, c, d) in enumerate(g) if j != k and np.hypot(a + c / 2 - cx, b + d / 2 - cy) < 1.5 * max(w, h))
        G.append(dict(size=np.sqrt(w * h), sharp=cv2.Laplacian(crop, cv2.CV_32F).var() if crop.size else 0,
                      contrast=crop.std() if crop.size else 0, colorsep=np.linalg.norm(L_in - L_out), bright=crop.mean() if crop.size else 0,
                      edge=int(x < 5 or y < 5 or x + w > 1275 or y + h > 715), crowd=int(crowd > 0),
                      video=meta[iid]["video_id"], seq=meta[iid]["video_id"] + "/" + meta[iid]["sequence"],
                      hit=int(not free[k]), loc=int(free[k] and best[k] >= 0.3)))
    for i in np.flatnonzero(~tp):
        bi = iou_matrix(p[i:i+1, 1:], g).max() if len(g) else 0
        F.append(dict(conf=p[i, 0], empty=int(len(g) == 0), kind="loose/dup (IoU 0.1-0.5 w/ GT)" if bi >= 0.1 else "background",
                      seq=meta[iid]["video_id"] + "/" + meta[iid]["sequence"], size=np.sqrt(p[i, 3] * p[i, 4])))
n = len(G); hits = sum(r["hit"] for r in G); print(f"GT {n}, hit {hits}, missed {n-hits} (of which {sum(r['loc'] for r in G)} had a box at IoU 0.3-0.5 = localisation), FP {len(F)}")
def table(key, edges=None, labels=None):
    v = np.array([r[key] for r in G]); hit = np.array([r["hit"] for r in G]); loc = np.array([r["loc"] for r in G]); sz = np.array([r["size"] for r in G])
    if edges is None:
        cats = sorted(set(v)); idx = [v == c for c in cats]; names = [str(c) for c in cats]
    else:
        idx = [(v >= edges[i]) & (v < edges[i + 1]) for i in range(len(edges) - 1)]; names = labels
    print(f"\n{key}:"); print(f"  {'bin':22s} {'n':>4s} {'miss%':>6s} {'loc%':>5s} {'med size':>8s}")
    for nm, m in zip(names, idx):
        if m.sum(): print(f"  {nm:22s} {m.sum():4d} {100*(1-hit[m].mean()):6.1f} {100*loc[m].mean():5.1f} {np.median(sz[m]):8.1f}")
def q(key): v = np.array([r[key] for r in G]); e = np.quantile(v, [0, 1/3, 2/3, 1]); e[-1] += 1e-6; return e, [f"{a:.1f}-{b:.1f}" for a, b in zip(e[:-1], e[1:])]
table("size", [0, 25, 35, 50, 80, 1e9], ["<25", "25-35", "35-50", "50-80", ">=80"])
for k in ["sharp", "contrast", "colorsep", "bright"]:
    e, l = q(k); table(k, e, ["low " + l[0], "mid " + l[1], "high " + l[2]])
table("edge"); table("crowd"); table("video"); table("seq")
# logistic regression of miss on standardised log features, to separate confounds
import numpy.linalg as la
keys = ["size", "sharp", "contrast", "colorsep", "bright"]
X = np.array([[np.log1p(r[k]) for k in keys] + [r["edge"], r["crowd"]] for r in G]); X = (X - X.mean(0)) / X.std(0)
y = 1 - np.array([r["hit"] for r in G]); Xb = np.column_stack([np.ones(len(X)), X]); b = np.zeros(Xb.shape[1])
for _ in range(50):
    pr = 1 / (1 + np.exp(-Xb @ b)); W = pr * (1 - pr); H = Xb.T @ (Xb * W[:, None]) + 1e-6 * np.eye(len(b)); b += la.solve(H, Xb.T @ (y - pr))
se = np.sqrt(np.diag(la.inv(H)))
print("\nlogistic regression, P(miss) ~ standardised features (log for continuous). odds ratio per +1 sd, z:")
for nm, bb, s in zip(["size", "sharp", "contrast", "colorsep", "bright", "edge", "crowd"], b[1:], se[1:]):
    print(f"  {nm:9s} OR {np.exp(bb):5.2f}  z {bb/s:6.2f}")
print("\ncorrelation with log size:", {k: round(np.corrcoef(np.log1p([r[k] for r in G]), np.log([r['size'] for r in G]))[0, 1], 2) for k in ["sharp", "contrast", "colorsep", "bright"]})
# FP tables
print(f"\nFP ({len(F)}):")
for key in ["kind", "empty"]:
    v = [r[key] for r in F]
    for c in sorted(set(v)): print(f"  {key}={c}: {v.count(c)}")
print("  FP by sequence:", {s: [r['seq'] for r in F].count(s) for s in sorted(set(r['seq'] for r in F))})
c = np.array([r["conf"] for r in F]); print("  FP conf: 0.21-0.3", ((c < 0.3)).sum(), " 0.3-0.5", ((c >= 0.3) & (c < 0.5)).sum(), " >=0.5", (c >= 0.5).sum())
