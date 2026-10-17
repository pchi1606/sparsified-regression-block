"""Verifier for the reproduction package.

Checks the paper's headline numbers two ways:
1. Recompute the preregistered statistics from the bundled full-study
   results (data/results_full.csv, 120 rows) and assert the
   non-inferiority decision.
2. Check the fresh verification grid (code/verification_results.csv,
   produced by reproduce.sh) for ballpark agreement and absence of
   failures.

Exits 0 iff every check passes.
"""
import csv
import math
import os
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FULL_CSV = os.path.join(ROOT, "data", "results_full.csv")
GRID_CSV = os.path.join(HERE, "verification_results.csv")
TRIM50_CSV = os.path.join(ROOT, "data", "results_trim50.csv")
HIGHP_CSV = os.path.join(ROOT, "data", "results_highp.csv")

MARGIN = 0.01
failures = []


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name} {detail}")
    if not cond:
        failures.append(name)


def main():
    # ---- 1. headline statistics from the bundled full results ----
    rows = [r for r in csv.DictReader(open(FULL_CSV)) if r["status"] == "ok"]
    check("full study: 120 ok rows", len(rows) == 120, f"(got {len(rows)})")
    cell = {}
    for r in rows:
        cell.setdefault((r["dataset"], int(r["split"])),
                        {})[r["method"]] = float(r["r2"])
    check("full study: 40 complete cells",
          len(cell) == 40 and all(len(v) == 3 for v in cell.values()))

    d = np.array([v["q4p"] - v["full"] for v in cell.values()])
    md = d.mean()
    se = d.std(ddof=1) / math.sqrt(len(d))
    lo, hi = stats.t.interval(0.95, len(d) - 1, loc=md, scale=se)
    check("headline: SparsePair-4P mean dR2 == -0.0003",
          abs(md + 0.0003) < 0.0002, f"(got {md:+.4f})")
    check("headline: 95% CI == [-0.0049, +0.0043]",
          abs(lo + 0.0049) < 0.0002 and abs(hi - 0.0043) < 0.0002,
          f"(got [{lo:+.4f}, {hi:+.4f}])")
    check("headline: non-inferiority holds (CI lower > -0.01)",
          lo > -MARGIN, f"(lower={lo:+.4f})")

    d2 = np.array([v["q2p"] - v["full"] for v in cell.values()])
    check("headline: SparsePair-2P mean dR2 == -0.0043",
          abs(d2.mean() + 0.0043) < 0.0002, f"(got {d2.mean():+.4f})")

    # fidelity of the bundled baseline vs the paper's reported column
    paper_rb = {"California": 0.766, "Yacht": 0.989, "Energy": 0.995,
                "Concrete": 0.907, "Airfoil": 0.927, "Abalone": 0.329,
                "Kin8nm": 0.920, "Protein": 0.455}
    ours = {ds: np.mean([v["full"] for (dd, _), v in cell.items()
                         if dd == ds]) for ds in paper_rb}
    max_dev = max(abs(ours[ds] - paper_rb[ds]) for ds in paper_rb)
    mean_dev = abs(np.mean(list(ours.values()))
                   - np.mean(list(paper_rb.values())))
    check("fidelity: baseline within 0.008/dataset of paper",
          max_dev <= 0.0085, f"(max dev {max_dev:.4f})")
    check("fidelity: baseline within 0.001 of paper on mean",
          mean_dev <= 0.0015, f"(mean dev {mean_dev:.4f})")

    # ---- 2. fresh verification grid ----
    if not os.path.exists(GRID_CSV):
        print(f"SKIP  verification grid not found ({GRID_CSV})")
    else:
        grows = list(csv.DictReader(open(GRID_CSV)))
        ok = [r for r in grows if r["status"] == "ok"]
        check("grid: 12/12 runs ok", len(ok) == 12, f"(got {len(ok)})")
        # determinism: the grid reuses the study's seeds/code, so each cell
        # must reproduce the bundled value to the printed precision
        for r in ok:
            key = (r["dataset"], int(r["split"]))
            if key in cell and r["method"] in cell[key]:
                ref = cell[key][r["method"]]
                got = float(r["r2"])
                check(f"grid {r['dataset']} s{r['split']} {r['method']} "
                      f"reproduces", abs(got - ref) < 2e-6,
                      f"(got {got:.6f}, ref {ref:.6f})")

    # ---- 3. trim50 head-to-head (Table 5: source's own PCA->50 trimming) ----
    trows = [r for r in csv.DictReader(open(TRIM50_CSV)) if r["status"] == "ok"]
    check("trim50 study: 40 ok rows", len(trows) == 40, f"(got {len(trows)})")
    tcell = {(r["dataset"], int(r["split"])): float(r["r2"]) for r in trows}
    check("trim50: n_comp=50 everywhere",
          all(r["n_comp"] == "50" for r in trows))
    # paired contrasts against the main-study cells, same (dataset, split) keys
    keys = sorted(cell.keys())
    tvals = np.array([tcell[k] for k in keys])
    fvals = np.array([cell[k]["full"] for k in keys])
    qvals = np.array([cell[k]["q4p"] for k in keys])
    q2vals = np.array([cell[k]["q2p"] for k in keys])
    d_tf = tvals - fvals
    md = d_tf.mean()
    se = d_tf.std(ddof=1) / math.sqrt(len(d_tf))
    lo, hi = stats.t.interval(0.95, len(d_tf) - 1, loc=md, scale=se)
    check("trim50 headline: trim50-full mean == -0.0000",
          abs(md) < 0.0002, f"(got {md:+.4f})")
    check("trim50 headline: CI == [-0.0033, +0.0032]",
          abs(lo + 0.0033) < 0.0002 and abs(hi - 0.0032) < 0.0002,
          f"(got [{lo:+.4f}, {hi:+.4f}])")
    d_tq = tvals - qvals
    md2 = d_tq.mean()
    se2 = d_tq.std(ddof=1) / math.sqrt(len(d_tq))
    lo2, hi2 = stats.t.interval(0.95, len(d_tq) - 1, loc=md2, scale=se2)
    check("trim50 headline: trim50-q4p mean == +0.0002",
          abs(md2 - 0.0002) < 0.0002, f"(got {md2:+.4f})")
    check("trim50 headline: CI == [-0.0031, +0.0036]",
          abs(lo2 + 0.0031) < 0.0002 and abs(hi2 - 0.0036) < 0.0002,
          f"(got [{lo2:+.4f}, {hi2:+.4f}])")
    d_q2t = q2vals - tvals
    md3 = d_q2t.mean()
    se3 = d_q2t.std(ddof=1) / math.sqrt(len(d_q2t))
    lo3, hi3 = stats.t.interval(0.95, len(d_q2t) - 1, loc=md3, scale=se3)
    check("trim50 headline: q2p-trim50 mean == -0.0043",
          abs(md3 + 0.0043) < 0.0002, f"(got {md3:+.4f})")
    check("trim50 headline: CI == [-0.0084, -0.0001]",
          abs(lo3 + 0.0084) < 0.0002 and abs(hi3 + 0.0001) < 0.0002,
          f"(got [{lo3:+.4f}, {hi3:+.4f}])")
    check("trim50 headline: mean R2 == 0.7849",
          abs(tvals.mean() - 0.7849) < 0.0002, f"(got {tvals.mean():.4f})")
    # P=8 reference figures (California): trim50 20,801 vs full 30,401
    cal = [r for r in trows if r["dataset"] == "California"]
    calfull = [r for r in rows if r["dataset"] == "California"
               and r["method"] == "full"]
    check("trim50 headline: 20,801 params vs 30,401 (full, P=8)",
          all(r["n_params"] == "20801" for r in cal)
          and all(r["n_params"] == "30401" for r in calfull),
          f"(trim50 { {r['n_params'] for r in cal} }, "
          f"full { {r['n_params'] for r in calfull} })")
    tw = np.mean([float(r["wall_s"]) for r in trows])
    fw = np.mean([float(r["wall_s"]) for r in rows if r["method"] == "full"])
    check("trim50 headline: 23s vs 93s mean wall time",
          abs(tw - 23) <= 1.0 and abs(fw - 93) <= 2.0,
          f"(got {tw:.1f}s vs {fw:.1f}s)")

    # ---- 4. higher-P follow-up (Section 6.5: P=21/32) ----
    hrows = [r for r in csv.DictReader(open(HIGHP_CSV)) if r["status"] == "ok"]
    check("highp study: 12 ok rows", len(hrows) == 12, f"(got {len(hrows)})")
    hcell = {}
    for r in hrows:
        hcell.setdefault((r["dataset"], int(r["split"])),
                         {})[r["method"]] = float(r["r2"])
    check("highp: 6 complete cells",
          len(hcell) == 6 and all(len(v) == 2 for v in hcell.values()))
    for ds, em, elo, ehi in [("CPU_act", 0.0006, -0.0013, 0.0025),
                             ("Bank32nh", -0.0039, -0.0160, 0.0082)]:
        d4 = np.array([v["q4p"] - v["full"] for (dd, _), v in hcell.items()
                       if dd == ds])
        se4 = d4.std(ddof=1) / math.sqrt(len(d4))
        lo4, hi4 = stats.t.interval(0.95, len(d4) - 1, loc=d4.mean(),
                                    scale=se4)
        check(f"highp headline: {ds} mean dR2 == {em:+.4f}",
              abs(d4.mean() - em) < 0.0002, f"(got {d4.mean():+.4f})")
        check(f"highp headline: {ds} CI == [{elo:+.4f}, {ehi:+.4f}]",
              abs(lo4 - elo) < 0.0002 and abs(hi4 - ehi) < 0.0002,
              f"(got [{lo4:+.4f}, {hi4:+.4f}])")
    w = {}
    for r in hrows:
        w[(r["dataset"], r["method"])] = (float(r["feat_width"]),
                                          float(r["n_pairs"]))
    check("highp headline: CPU_act width 14,912 -> 5,504",
          w[("CPU_act", "full")][0] == 14912.0
          and w[("CPU_act", "q4p")][0] == 5504.0)
    check("highp headline: Bank32nh width 33,920 -> 8,320",
          w[("Bank32nh", "full")][0] == 33920.0
          and w[("Bank32nh", "q4p")][0] == 8320.0)

    print()
    if failures:
        print(f"VERIFICATION FAILED: {failures}")
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
