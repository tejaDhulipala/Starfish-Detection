"""model_dev/index.csv: one row per config. Training adds the row, evaluation fills in the best F2."""

import csv

from src.paths import MODEL_DEV

INDEX = MODEL_DEV / "index.csv"
FIELDS = ["config", "description", "best_f2", "best_conf", "best_run", "best_weights"]


def read_index():
    if not INDEX.exists():
        return []
    with open(INDEX, newline="") as f:
        return list(csv.DictReader(f))


def upsert(config, **fields):
    rows = read_index()
    row = next((r for r in rows if r["config"] == config), None)
    if row is None:
        row = {k: "" for k in FIELDS} | {"config": config}
        rows.append(row)
    row.update({k: str(v) for k, v in fields.items()})
    rows.sort(key=lambda r: r["config"])
    INDEX.parent.mkdir(exist_ok=True)
    with open(INDEX, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def record_f2(config, f2, conf, run, weights):
    """Update the config's row only if this F2 beats the recorded best."""
    row = next((r for r in read_index() if r["config"] == config), None)
    best = float(row["best_f2"]) if row and row["best_f2"] else -1.0
    if f2 > best:
        upsert(config, best_f2=f"{f2:.4f}", best_conf=f"{conf:.2f}", best_run=run, best_weights=weights)
        return True
    return False
