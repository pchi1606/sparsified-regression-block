#!/usr/bin/env bash
# One-command reproduction for "Diagonal Redundancy, Not Sparsity, in the Regression Block".
# Runs the verification grid (12 trained models, CPU-only) and then the
# headline verifier. Exits non-zero if any check fails.
set -euo pipefail
cd "$(dirname "$0")"
python3 -c "import numpy, sklearn, scipy, pandas, matplotlib" \
  || { echo "missing dependency: numpy scikit-learn scipy pandas matplotlib"; exit 1; }
python3 code/run_verification.py
python3 code/verify_headlines.py
