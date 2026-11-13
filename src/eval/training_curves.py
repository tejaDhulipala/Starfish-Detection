"""Plot columns of an Ultralytics results.csv (works on running, stopped and finished runs).

- Same range for every curve (default): each series is min-max scaled to 0..1 on one shared
  axis, so a loss and F2 are equally readable. The legend keeps each series' real range.
  Shapes and trends compare across series; heights do not mean anything across series.
- --skip K (default 1): the first K epochs are left out of each series' 0..1 scale. Warmup
  epochs can be huge outliers (c01's epoch-1 val cls loss is 13.7) and would flatten a series;
  they may fall off the plot.
- --separate: real units, one panel per quantity (train and val of the same loss share a panel).
- Color = quantity, line style = split: solid for train and metrics, dashed for val losses.
- F2 moving average (--ma N, 0 turns it off): trailing mean of the last N epochs (fewer at the
  start), scaled with raw F2's own range so the two stay comparable.
- NaN entries (e.g. a val loss that overflowed) are left as gaps and ignored for scaling.

RUN is a results.csv, a run folder (model_dev/<config>/seed0), a config folder (uses seed0),
or a config name or prefix (c08). Saves a PNG next to results.csv unless --out or --show.

Usage (from the repo root):
    python -m src.eval.training_curves c08
    python -m src.eval.training_curves c08 --preset metrics --separate
    python -m src.eval.training_curves c06 --cols metrics/F2 train/box_loss val/box_loss --show
"""

import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np

from src.paths import ROOT

# Fixed color per quantity (reference categorical palette, light mode), so a column keeps
# its color whatever else is selected. Quantities past the 8 hues get neutral gray + their own dash.
SERIES_STYLE = {
    "box_loss":     ("#2a78d6", None),
    "cls_loss":     ("#eb6834", None),
    "l1_loss":      ("#1baf7a", None),
    "F2":           ("#6250d6", None),
    "recall(B)":    ("#008300", None),
    "precision(B)": ("#e87ba4", None),
    "mAP50(B)":     ("#eda100", None),
    "mAP50-95(B)":  ("#e34948", None),
    "F2_conf":      ("#898781", (0, (1, 1.5))),
    "time":         ("#898781", (0, (4, 2, 1, 2))),
}
LR_STYLE = ("#898781", (0, (6, 2)))
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"

PRESETS = {
    "default": ["train/box_loss", "val/box_loss", "train/cls_loss", "val/cls_loss", "metrics/F2"],
    "losses": ["train/box_loss", "val/box_loss", "train/cls_loss", "val/cls_loss", "train/l1_loss", "val/l1_loss"],
    "metrics": ["metrics/F2", "metrics/recall(B)", "metrics/precision(B)", "metrics/mAP50(B)"],
}


def resolve_results(run):
    """Path to results.csv from a file, run folder, config folder or config name prefix."""
    p = Path(run)
    candidates = [p, ROOT / p]
    matches = sorted((ROOT / "model_dev").glob(f"{run}*")) if "/" not in run else []
    if len(matches) > 1:
        raise SystemExit(f"{run!r} matches several configs: {', '.join(m.name for m in matches)}")
    candidates += matches
    for c in candidates:
        for f in (c, c / "results.csv", c / "seed0" / "results.csv"):
            if f.is_file():
                return f
    raise SystemExit(f"no results.csv found for {run!r}")


def load_results(path):
    with open(path, newline="") as f:
        rows = [{k.strip(): v.strip() for k, v in r.items()} for r in csv.DictReader(f)]
    return {k: np.array([float(r[k]) if r[k] else np.nan for r in rows]) for k in rows[0]} if rows else {}


def quantity(col):
    return col.split("/", 1)[-1]


def style(col):
    if col.startswith("lr/"):
        color, dash = LR_STYLE
    else:
        color, dash = SERIES_STYLE.get(quantity(col), ("#898781", (0, (2, 2))))
    if col.startswith("val/"):
        dash = (0, (5, 2.5))
    return color, (dash or "solid")


def trailing_mean(y, n):
    """Mean of the last n finite values up to each epoch (NaNs skipped)."""
    ok = np.isfinite(y)
    c = np.cumsum(np.insert(np.where(ok, y, 0.0), 0, 0.0))
    k = np.cumsum(np.insert(ok, 0, 0))
    idx = np.arange(1, len(y) + 1)
    lo = np.maximum(idx - n, 0)
    cnt = k[idx] - k[lo]
    return np.where(cnt > 0, (c[idx] - c[lo]) / np.maximum(cnt, 1), np.nan)


def scale01(y, lo, hi):
    return (y - lo) / (hi - lo) if hi > lo else np.full_like(y, 0.5)


def plot_results(path, columns, normalize=True, separate=False, ma_window=None, mark_best=True, skip=1):
    """Return the figure, or None if there is nothing to plot."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    data = load_results(path)
    if not data:
        print(f"{path}: no epochs yet")
        return None
    ep = data["epoch"]
    has_f2 = "metrics/F2" in data
    missing = [c for c in columns if c not in data]
    if missing:
        print(f"not in {path.name}, skipped: {', '.join(missing)}")
    series = [(c, data[c], c, None) for c in columns if c in data]  # (label, y, style key, scale source)
    if ma_window and has_f2:
        series.append((f"F2, mean of last {ma_window}", trailing_mean(data["metrics/F2"], ma_window), "metrics/F2", "metrics/F2"))
    if not series:
        print("no columns to plot")
        return None
    best = int(np.nanargmax(data["metrics/F2"])) if has_f2 and np.isfinite(data["metrics/F2"]).any() else None

    plt.rcParams.update({"axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
                         "text.color": INK, "font.size": 10})
    groups = {}
    for s in series:
        groups.setdefault(quantity(s[2]) if separate else "all", []).append(s)
    n = len(groups)
    ncols = 1 if n == 1 else 2
    nrows = -(-n // ncols)
    width = 11 if ncols == 2 else (12 if normalize and not separate else 9)
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, 3.6 * nrows if separate else 5),
                             sharex=True, squeeze=False, facecolor=SURFACE)
    for ax in axes.flat[n:]:
        ax.set_visible(False)

    shared = normalize and not separate
    for ax, (gname, group) in zip(axes.flat, groups.items()):
        ax.set_facecolor(SURFACE)
        for label, y, key, scale_src in group:
            color, dash = style(key)
            src = data[scale_src] if scale_src is not None else y
            ref = src[skip:] if (shared and len(src) > skip) else src
            ref = ref[np.isfinite(ref)]
            lo, hi = (float(ref.min()), float(ref.max())) if ref.size else (0.0, 1.0)
            is_ma = label.startswith("F2, mean")
            yy = scale01(y, lo, hi) if shared else y
            legend = label
            if shared:
                legend += "  (scaled with F2)" if is_ma else f"  ({lo:.4g} to {hi:.4g})"
            n_nan = int((~np.isfinite(y)).sum())
            if n_nan and not is_ma:
                legend += f"  [{n_nan} NaN]"
            ax.plot(ep, yy, color=color, linestyle=dash, linewidth=3 if is_ma else 2,
                    alpha=1.0 if (is_ma or not ma_window or key != "metrics/F2") else 0.45, label=legend)
        if mark_best and best is not None:
            ax.axvline(ep[best], color=INK_2, linewidth=1, linestyle=(0, (2, 3)))
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        if separate:
            ax.set_title(gname, color=INK, fontsize=10, loc="left")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        if shared:
            ax.legend(frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(1.01, 1))
        else:
            ax.legend(frameon=False, fontsize=8.5, loc="best")
    for ax in axes[-1]:
        ax.set_xlabel("epoch")
    if shared:
        axes[0, 0].set_ylabel("scaled per series (0 = its min, 1 = its max)" + (f", epochs > {skip}" if skip else ""))
        axes[0, 0].set_ylim(-0.05, 1.05)

    run = Path(path).parent
    title = f"{run.parent.name} / {run.name}  ({len(ep)} epochs)"
    if best is not None:
        title += f"   best F2 {data['metrics/F2'][best]:.3f} at epoch {int(ep[best])}" + (" (dotted line)" if mark_best else "")
    fig.suptitle(title, color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="results.csv, run folder, config folder, or config name prefix (e.g. c08)")
    ap.add_argument("--cols", nargs="+", help="columns to plot (overrides --preset)")
    ap.add_argument("--preset", choices=PRESETS, default="default", help="column set: default, losses (overfitting), metrics")
    ap.add_argument("--list", action="store_true", help="print the available columns and exit")
    ap.add_argument("--raw", action="store_true", help="real units on one axis instead of scaling each series to 0..1")
    ap.add_argument("--separate", action="store_true", help="one panel per quantity, real units")
    ap.add_argument("--ma", type=int, default=5, help="F2 moving-average window in epochs (0 = off)")
    ap.add_argument("--no-best", action="store_true", help="don't mark the best-F2 epoch")
    ap.add_argument("--skip", type=int, default=1, help="epochs left out of each series' 0..1 scale")
    ap.add_argument("--out", type=Path, help="PNG path (default: training_curves.png next to results.csv)")
    ap.add_argument("--show", action="store_true", help="open a window instead of saving")
    args = ap.parse_args()

    path = resolve_results(args.run)
    if args.list:
        print("\n".join(c for c in load_results(path) if c != "epoch"))
        return
    if not args.show:
        matplotlib.use("Agg")
    fig = plot_results(path, args.cols or PRESETS[args.preset], normalize=not args.raw, separate=args.separate,
                       ma_window=args.ma or None, mark_best=not args.no_best, skip=args.skip)
    if fig is None:
        return
    if args.show:
        import matplotlib.pyplot as plt
        plt.show()
    else:
        out = args.out or path.parent / "training_curves.png"
        fig.savefig(out, dpi=130, facecolor=fig.get_facecolor(), bbox_inches="tight")
        print(out)


if __name__ == "__main__":
    main()
