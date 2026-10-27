#!/usr/bin/env bash
# Runs the exploratory-analysis tests that need no data.
# The tests listed in replication/deselected-tests.txt need withheld data, git history
# or the computing sites' job scripts, and are skipped here.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
args=()
while IFS= read -r line; do
  case "$line" in ''|'#'*) continue ;; esac
  args+=(--deselect "$line")
done < replication/deselected-tests.txt
exec uv run --frozen python -m pytest -p no:cacheprovider -q "${args[@]}" results/e189-exploratory-20261022/tests "$@"
