#!/usr/bin/env bash

set -e
cd "$(dirname "$0")/../.."

export DAYCENT_ROOT="${DAYCENT_ROOT:-$(cd .. && pwd)}"
export DAYCENT_OUTPUT_ROOT="${DAYCENT_OUTPUT_ROOT:-$DAYCENT_ROOT/output}"
export DAYCENT_PROCESSED_ROOT="${DAYCENT_PROCESSED_ROOT:-$DAYCENT_ROOT/data}"

python compare_selection_runs.py \
    --runs-dir "${RUNS_DIR:-$DAYCENT_OUTPUT_ROOT/selection}" \
    "$@"
