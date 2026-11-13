"""Interactive training curves: python -m src.eval.training_curves (from the repo root).

Opens a matplotlib window; everything is set from the controls on the right:
- Run: every model_dev/<config>/<seed>/results.csv (running, stopped or finished). Opens on the
  most recently updated one.
- Columns: any column of that results.csv; presets for the usual sets.
- Same range for every curve (default on): each series is min-max scaled to 0..1 on one shared
  axis, so a loss and F2 are equally readable. The legend keeps each series' real range.
  Shapes and trends compare across series; heights do not mean anything across series.
- Scale ignores first K epochs (default 1): warmup epochs can be huge outliers (c01's epoch-1
  val cls loss is 13.7) and would flatten a series; they may fall off the plot.
- Separate panels: real units, one panel per quantity (train and val of a loss share a panel).
- Color = quantity, line style = split: solid for train and metrics, dashed for val losses.
- F2 moving average: trailing mean of the last N epochs (fewer at the start), scaled with raw
  F2's own range so the two stay comparable.
- NaN cells (e.g. a val loss that overflowed) are gaps; the legend counts them.
- Reload picks up new epochs; Auto reload does it every 30 s.
- Save PNG writes the plot (without the controls) to training_curves.png next to results.csv.
"""

import csv

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider

from src.paths import MODEL_DEV

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
    "Default": {"train/box_loss", "val/box_loss", "train/cls_loss", "val/cls_loss", "metrics/F2"},
    "Losses": {"train/box_loss", "val/box_loss", "train/cls_loss", "val/cls_loss", "train/l1_loss", "val/l1_loss"},
    "Metrics": {"metrics/F2", "metrics/recall(B)", "metrics/precision(B)", "metrics/mAP50(B)"},
}
OPTIONS = ["Same range for every curve", "Separate panels (real units)", "F2 moving average", "Mark best F2 epoch",
           "Auto reload (30 s)"]
OPTION_DEFAULTS = [True, False, True, True, False]
PLOT_AREA = (0.05, 0.08, 0.55, 0.92)  # left, bottom, right, top; the shared-axis legend sits right of it
PLOT_AREA_SEPARATE = (0.05, 0.08, 0.72, 0.92)
SIDEBAR_X, SIDEBAR_W = 0.755, 0.235

plt.rcParams.update({"axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
                     "text.color": INK, "font.size": 10})


# ---------------------------------------------------------------- data

def find_runs():
    return sorted(MODEL_DEV.glob("*/*/results.csv"))


def run_label(path):
    cfg, seed = path.parent.parent.name, path.parent.name
    label = cfg if len(cfg) <= 32 else cfg[:31] + "…"
    return label if seed == "seed0" else f"{label} / {seed}"


def load_results(path):
    try:
        with open(path, newline="") as f:
            rows = [{k.strip(): (v or "").strip() for k, v in r.items() if k is not None} for r in csv.DictReader(f)]
    except OSError:
        return {}
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


# ---------------------------------------------------------------- plotting

def draw_curves(fig, path, data, columns, normalize=True, separate=False, ma_window=None, mark_best=True, skip=1,
                area=None):
    """Draw into fig inside area (left, bottom, right, top); returns the axes it made."""
    l, b, r, t = area or (PLOT_AREA if normalize and not separate else PLOT_AREA_SEPARATE)

    def message(text):
        ax = fig.add_axes([l, b, r - l, t - b])
        ax.axis("off")
        ax.text(0.5, 0.5, text, ha="center", va="center", color=INK_2, transform=ax.transAxes)
        fig.suptitle(run_label(path), color=INK, fontsize=11, x=0.01, ha="left")
        return [ax]

    if not data:
        return message("no epochs yet")
    ep = data["epoch"]
    has_f2 = "metrics/F2" in data
    series = [(c, data[c], c, None) for c in columns if c in data]  # (label, y, style key, scale source)
    if ma_window and has_f2:
        series.append((f"F2, mean of last {ma_window}", trailing_mean(data["metrics/F2"], ma_window), "metrics/F2", "metrics/F2"))
    if not series:
        return message("Select at least one column.")
    best = int(np.nanargmax(data["metrics/F2"])) if has_f2 and np.isfinite(data["metrics/F2"]).any() else None

    groups = {}
    for s in series:
        groups.setdefault(quantity(s[2]) if separate else "all", []).append(s)
    n = len(groups)
    ncols = 1 if n == 1 else 2
    nrows = -(-n // ncols)
    gs = fig.add_gridspec(nrows, ncols, left=l, right=r, bottom=b, top=t, hspace=0.35, wspace=0.18)

    shared = normalize and not separate
    axes = []
    for i, (gname, group) in enumerate(groups.items()):
        ax = fig.add_subplot(gs[i // ncols, i % ncols], sharex=axes[0] if axes else None)
        axes.append(ax)
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
        if i >= n - ncols:
            ax.set_xlabel("epoch")
    if shared:
        axes[0].set_ylabel("scaled per series (0 = its min, 1 = its max)" + (f", epochs > {skip}" if skip else ""))
        axes[0].set_ylim(-0.05, 1.05)

    title = f"{path.parent.parent.name} / {path.parent.name}  ({len(ep)} epochs)"
    if best is not None:
        title += f"   best F2 {data['metrics/F2'][best]:.3f} at epoch {int(ep[best])}" + (" (dotted line)" if mark_best else "")
    fig.suptitle(title, color=INK, fontsize=11, x=0.01, ha="left")
    return axes


# ---------------------------------------------------------------- window

class CurvesApp:
    def __init__(self):
        self.runs = find_runs()
        if not self.runs:
            raise SystemExit(f"no results.csv under {MODEL_DEV}")
        self.path = max(self.runs, key=lambda p: p.stat().st_mtime)
        self.data, self.mtime = {}, None
        self.selected = set(PRESETS["Default"])
        self.plot_axes, self.col_ax, self.col_check = [], None, None

        self.fig = plt.figure(figsize=(16, 9.5), facecolor=SURFACE)
        self.fig.canvas.manager.set_window_title("Training curves")
        x, w = SIDEBAR_X, SIDEBAR_W

        def heading(y, text):
            self.fig.text(x, y, text, fontsize=9, fontweight="bold", color=INK_2)

        heading(0.965, "Run")
        n_runs = len(self.runs)
        run_h = min(0.025 * n_runs, 0.26)
        self.run_ax = self.fig.add_axes([x, 0.955 - run_h, w, run_h], facecolor=SURFACE)
        self.run_radio = RadioButtons(self.run_ax, [run_label(p) for p in self.runs],
                                      active=self.runs.index(self.path), activecolor=SERIES_STYLE["F2"][0])
        self._small(self.run_radio)
        self.run_radio.on_clicked(self.on_run)

        self.col_top = 0.94 - run_h  # columns list hangs from here
        heading(self.col_top, "Columns")

        heading(0.285, "Options")
        opt_ax = self.fig.add_axes([x, 0.165, w, 0.115], facecolor=SURFACE)
        self.opts = CheckButtons(opt_ax, OPTIONS, OPTION_DEFAULTS)
        self._small(self.opts)
        self.opts.on_clicked(self.on_option)

        self.ma = Slider(self.fig.add_axes([x + 0.09, 0.135, w - 0.13, 0.02]), "F2 average: last N", 2, 20,
                         valinit=5, valstep=1, color=SERIES_STYLE["F2"][0])
        self.skip = Slider(self.fig.add_axes([x + 0.09, 0.105, w - 0.13, 0.02]), "Scale ignores first K", 0, 10,
                           valinit=1, valstep=1, color=INK_2)
        for s in (self.ma, self.skip):
            s.label.set_fontsize(8)
            s.valtext.set_fontsize(8)
            s.on_changed(lambda _: self.redraw())

        self.buttons = []
        bw = (w - 0.02) / 3
        for i, name in enumerate(PRESETS):
            self._button([x + i * (bw + 0.01), 0.055, bw, 0.035], name, lambda _, n=name: self.on_preset(n))
        bw2 = (w - 0.01) / 2
        self._button([x, 0.012, bw2, 0.035], "Reload", lambda _: self.reload(force=True))
        self._button([x + bw2 + 0.01, 0.012, bw2, 0.035], "Save PNG", lambda _: self.save())
        self.status = self.fig.text(0.05, 0.012, "", fontsize=8, color=INK_2)

        self.timer = self.fig.canvas.new_timer(interval=30_000)
        self.timer.add_callback(self.on_timer)
        self.timer.start()
        self.reload(force=True)

    @staticmethod
    def _small(widget):
        for t in widget.labels:
            t.set_fontsize(8)

    def _button(self, rect, label, cb):
        b = Button(self.fig.add_axes(rect), label, color="#efeeea", hovercolor=GRID)
        b.label.set_fontsize(8)
        b.on_clicked(cb)
        self.buttons.append(b)  # widgets must stay referenced to keep working

    # --- state

    def option(self, name):
        return self.opts.get_status()[OPTIONS.index(name)]

    def columns(self):
        return [c for c in self.data if c != "epoch"]

    def build_columns(self):
        """(Re)build the column checkboxes for the current run, keeping the selection."""
        if self.col_ax is not None:
            self.col_ax.remove()
        cols = self.columns()
        bottom = 0.30
        h = min(0.022 * max(len(cols), 1), self.col_top - 0.01 - bottom)
        self.col_ax = self.fig.add_axes([SIDEBAR_X, self.col_top - 0.01 - h, SIDEBAR_W, h], facecolor=SURFACE)
        if not cols:
            self.col_ax.axis("off")
            self.col_check = None
            return
        self.col_check = CheckButtons(self.col_ax, cols, [c in self.selected for c in cols])
        self._small(self.col_check)
        self.col_check.on_clicked(self.on_column)
        self._col_names = cols

    def reload(self, force=False):
        try:
            mtime = self.path.stat().st_mtime
        except OSError:
            mtime = None
        if not force and mtime == self.mtime:
            return
        old_cols = self.columns()
        self.data, self.mtime = load_results(self.path), mtime
        if force or self.columns() != old_cols:
            self.build_columns()
        self.redraw()

    def redraw(self):
        for ax in self.plot_axes:
            ax.remove()
        self.plot_axes = draw_curves(self.fig, self.path, self.data, self.plot_columns(), **self.plot_kwargs())
        self.fig.canvas.draw_idle()

    def plot_columns(self):
        return [c for c in self.columns() if c in self.selected]

    def plot_kwargs(self):
        return dict(normalize=self.option(OPTIONS[0]), separate=self.option(OPTIONS[1]),
                    ma_window=int(self.ma.val) if self.option(OPTIONS[2]) else None,
                    mark_best=self.option(OPTIONS[3]), skip=int(self.skip.val))

    # --- callbacks

    def on_run(self, label):
        self.path = self.runs[[run_label(p) for p in self.runs].index(label)]
        self.mtime = None
        self.reload(force=True)
        self.status.set_text("")

    def on_column(self, label):
        self.selected ^= {label}
        self.redraw()

    def on_option(self, _):
        self.redraw()

    def on_preset(self, name):
        self.selected = set(PRESETS[name])
        self.build_columns()
        self.redraw()

    def on_timer(self):
        if self.option("Auto reload (30 s)"):
            self.reload()

    def save(self):
        kw = self.plot_kwargs()
        n_panels = len({quantity(c) for c in self.plot_columns()} | ({"F2"} if kw["ma_window"] else set()))
        separate = kw["separate"]
        nrows = -(-n_panels // (1 if n_panels == 1 else 2)) if separate else 1
        fig = Figure(figsize=(11, 3.6 * nrows) if separate else (12, 5), facecolor=SURFACE)
        area = (0.07, 0.1, 0.97, 0.88) if separate or not kw["normalize"] else (0.07, 0.1, 0.68, 0.88)
        draw_curves(fig, self.path, self.data, self.plot_columns(), area=area, **kw)
        out = self.path.parent / "training_curves.png"
        fig.savefig(out, dpi=130, facecolor=SURFACE, bbox_inches="tight")
        self.status.set_text(f"saved {out}")
        print(f"saved {out}")
        self.fig.canvas.draw_idle()


def main():
    app = CurvesApp()  # noqa: F841  (keeps the widgets alive while the window is open)
    plt.show()


if __name__ == "__main__":
    main()
