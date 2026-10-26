#!/usr/bin/env bash
# How the paper's numbers are produced from the recorded runs. The run data and the analysis
# inputs follow in step 2 of the package; until then this script prints the steps and exits.
set -euo pipefail
cd "$(dirname "$0")/.."
cat <<'STEPS'
Steps (run from the repository root, after `uv sync --frozen`):

1. Replay check of every run directory OUT (the gate before any analysis):
     uv run python scripts/verify_batch.py OUT --sources-dir SOURCES --hierarchy HIERARCHY

2. Confirmatory analysis (pre-registered), single-threaded:
     OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run python scripts/analyse.py \
       --batch OUT [--batch OUT ...] --questions QUESTIONS.jsonl --split test --n-reps 3 \
       --prereg-config data/analysis/prereg-config.json --bank-manifest BANK/manifest.json \
       [--ledger LEDGER.jsonl ...] --out DIR
   It writes DIR/results.json, DIR/numbers.tex and DIR/numbers.json.

3. Exploratory interval estimates (the repository root is the frozen code root):
     uv run python -I results/e189-exploratory-20261022/exploratory.py clusters --code-root . --out-dir clusters
     OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run python -I results/e189-exploratory-20261022/exploratory.py \
       analyse --code-root . --graph G --split test --batch OUT [--batch OUT ...] \
       --questions Q.jsonl --voided VOIDED.jsonl --out DIR \
       [--partition-record RECORD.json] [--ledger LEDGER.jsonl ...] [--store-proof CHECK.json ...]
   See the docstring of exploratory.py for every option.

Inputs that come with the data (step 2): the run directories (OUT), the question files, the
voided-run records, the partition records, the checker ledgers, the fault-bank manifest, the
frozen analysis inputs that data/analysis/prereg-config.json pins (calibration, cluster map,
multiplicity tables). The store-copy tables the exploratory analysis reads from its own
directory (copy-classes.json, class-overrides.json, extra-copies.json, copy-hosts.json,
endpoint-copies.json, store-proofs.json) are synthetic examples with the real schemas; a replicator who
reruns the experiments on their own infrastructure supplies these tables for their own store copies.
STEPS
