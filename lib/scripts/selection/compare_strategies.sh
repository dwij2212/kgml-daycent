#!/bin/bash
# ============================================================
# After running both sweeps, compare results across strategies.
#
# Usage:
#   bash scripts/compare_strategies.sh
# ============================================================
set -e
cd "$(dirname "$0")/.."

python compare_selection_runs.py \
    --runs-dir /users/6/mehta423/projects/daycent/output/selection
