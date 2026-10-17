# Reproduction package — "Diagonal Redundancy, Not Sparsity, in the Regression Block: A Preregistered Test and the Limits of Pair Subsampling"

This package lets an independent reader re-verify the paper's headline
results with one command:

```bash
bash reproduce.sh
```

## What it does

1. **Verification grid** (`code/run_verification.py`): trains 12 models
   CPU-only — 2 datasets (Yacht, Energy) x 2 splits x 3 methods
   (full quadratic baseline, SparsePair-4P, SparsePair-2P) — using the
   same seeds, splits, and code path as the full study. Writes
   `code/verification_results.csv`. Takes ~10 minutes on 2 CPUs.
   Datasets are fetched from OpenML on first run (public data) and
   cached locally.
2. **Headline verifier** (`code/verify_headlines.py`): recomputes the
   preregistered statistics from the bundled full-study results
   (`data/results_full.csv`, 120 rows: 8 datasets x 5 splits x 3
   methods) and asserts the non-inferiority decision; recomputes the
   trim50 head-to-head contrasts (`data/results_trim50.csv`, 40 rows:
   Table 5) and the higher-P follow-up results
   (`data/results_highp.csv`, 12 rows: P=21/32, Section 6.5); then
   checks each fresh grid cell reproduces its bundled value exactly
   (seeded determinism).

## Contents

- `reproduce.sh` — one-command entry point
- `code/rb_numpy.py` — NumPy implementation of the Regression Block and
  the SparsePair variants (no torch required)
- `code/run_verification.py`, `code/verify_headlines.py`,
  `code/data_loader.py` — verification harness
- `data/results_full.csv` — all 120 per-run results of the study
- `data/results_trim50.csv` — 40 per-run results of the trim50
  head-to-head (Table 5); `data/results_highp.csv` — 12 per-run
  results of the P=21/32 follow-up (Section 6.5)
- `paper.pdf` — the manuscript

## Requirements

Python 3.10+ with `numpy`, `scikit-learn`, `scipy`, `pandas`,
`matplotlib` (CPU only; no GPU, no paid services).

## Expected result

`ALL CHECKS PASSED`, including:

- SparsePair-4P minus full baseline: mean dR2 = -0.0003,
  95% CI [-0.0049, +0.0043] -> non-inferior at the preregistered
  0.01 margin;
- baseline reproduces the source paper's Regression Block column to
  0.001 on the mean (max 0.008 per dataset).

## License

MIT (see LICENSE).
