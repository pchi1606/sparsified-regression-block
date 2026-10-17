"""Verification grid for the reproduction package.

Trains a small but faithful slice of the preregistered protocol:
2 datasets (Yacht, Energy) x 2 splits x 3 methods (full, q4p, q2p)
= 12 runs. Uses the same seeds, splits, and code path as the full
study. Writes verification_results.csv next to this script.
"""
import csv
import os
import time

from rb_numpy import train_rb, r2_score
from data_loader import load_yacht_energy, prep_split

SEEDS = [42, 1042]
METHODS = ["full", "q4p", "q2p"]
OUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "verification_results.csv")


def main():
    datasets = load_yacht_energy()
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["dataset", "split", "seed", "method",
                           "r2", "epochs", "wall_s", "status"])
        writer.writeheader()
        for name, (X_full, y_full) in datasets.items():
            for si, seed in enumerate(SEEDS):
                Xtr, ytr, Xte, yte = prep_split(X_full, y_full, seed)
                for method in METHODS:
                    print(f"run {name} split{si} {method} ...", flush=True)
                    t0 = time.time()
                    try:
                        res = train_rb(Xtr, ytr, Xte, seed, mode=method)
                        r2 = r2_score(yte, res["y_pred"])
                        row = {"dataset": name, "split": si, "seed": seed,
                               "method": method, "r2": f"{r2:.6f}",
                               "epochs": res["epochs"],
                               "wall_s": f"{time.time() - t0:.1f}",
                               "status": "ok"}
                        print(f"  R2={r2:.4f}", flush=True)
                    except Exception as e:  # noqa: BLE001 - record and continue
                        row = {"dataset": name, "split": si, "seed": seed,
                               "method": method, "r2": "", "epochs": "",
                               "wall_s": f"{time.time() - t0:.1f}",
                               "status": f"FAILED: {e}"}
                        print(f"  FAILED: {e}", flush=True)
                    writer.writerow(row)
    print(f"wrote {OUT_CSV}", flush=True)


if __name__ == "__main__":
    main()
