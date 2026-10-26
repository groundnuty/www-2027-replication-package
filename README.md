# WWW 2027 replication package (code)

This repository holds the code behind the paper's experiments and reported numbers. It is anonymized for
double-blind review: names of people, institutions, computing sites, hosts and services are replaced by neutral
labels (see [Anonymization](#anonymization)).

The package is released in two steps:

1. **Code (this version):** the experiment framework at the frozen versions that produced the runs, and the analysis
   code that turns the recorded run outputs into the paper's numbers.
2. **Data (to follow):** the recorded run outputs and the analysis inputs (see [Data](#data)).

## Layout

The files keep the relative paths of the development repository, because the code locates its parts by those paths
(for example, the analysis checks the code tree it imports from).

| Path | What it is |
|---|---|
| `harness/` | The experiment framework: the agent harness and model providers (`harness/agent/`), run records and replay checks, the store and fault checks (`harness/checks/`, `harness/faults/`), the scorer (`harness/score/`), question generation (`harness/qgen*/`), sources and prompts (`harness/sources/`), the baselines (`harness/baselines/`), the checker loop (`harness/checkloop/`) and the pre-registered confirmatory analysis library (`harness/analysis/`). |
| `scripts/` | Entry points. `run_pilot.py` is the grid runner that produced the runs; `verify_batch.py` is the replay check every analysed run must pass; `relabel_harness_failure.py` sets aside an attempt that failed on infrastructure; `analyse.py` runs the confirmatory analysis (records to `results.json`, `numbers.tex`, `numbers.json`); `export_checker_ledger.py` and `checker_replay.py` export and replay the checker loop; the rest build questions, sources and prompts, calibrate the analysis, and probe models and stores. |
| `results/e167-admission/admit.py`, `results/e167-front/join.py` | The run-admission gate and the model-front provenance reader; the exploratory analysis imports them for its attribution check. |
| `results/e189-exploratory-20261022/` | The exploratory analysis: `exploratory.py` computes every exploratory interval estimate (cluster bootstrap, partitioned runs pooled as one union, store-copy and speed-class checks); `attribution.py` attributes model turns to served deployments; `copy_classes.py` builds the store-copy speed-class table; `driver/` holds the replay-matrix and look drivers; `tests/` its tests. The JSON tables beside them are the analysis's store-copy inputs (speed classes, class overrides, extra copies, copy hosts, endpoint copies, store proofs, admitted fingerprints), anonymized. |
| `data/freeze/frozen-config.json`, `data/analysis/prereg-config.json` | The frozen analysis configuration (seed, bootstrap size, models, thresholds, and the digests of the frozen inputs). |
| `tests/fixtures/analysis/` | Synthetic fixtures (toy questions and runs) used by the analysis tests. |
| `replication/versions/run-tags/*.patch` | One patch per other code version that runs were made with (see [Versions](#versions)). |
| `replication/versions/analysis/e150-route-and-setaside.patch` | The analysis changes of the parallel analysis line (see [Versions](#versions)). |
| `replication/run_tests.sh`, `replication/deselected-tests.txt` | Runs the analysis tests that need no data. |
| `replication/reproduce.sh` | How the analysis is run once the data of step 2 is in place. |
| `replication/PINS.md` | Integrity pins that anonymization changed: original and package values. |
| `replication/KNOWN-ISSUES.md` | Known issues: the checker's non-deterministic offline replay and how the analyses handled it. |
| `replication/DEVIATIONS.md` | Deviations from the pre-registration: what was not run, what changed after the freeze, and what is provisional. |

## Versions

Commit identifiers refer to the authors' development repository, which is private during review.

| Part | Version |
|---|---|
| Framework (`harness/`, `scripts/`, `results/e167-*`, `pyproject.toml`, `uv.lock`) | Tag `test-e180-20261022-sci-c30257bba485` (commit `e203098d`), the test runs' code (change-log entry 180 on top of entry 167). |
| Frozen analysis code | Commit `847cc275` (the config-fill head). Its framework paths equal tag `test-e167-20261022-sci-c30257bba485`; apply `replication/versions/run-tags/test-e167-20261022-sci-c30257bba485.patch` to obtain them. `harness/` is identical in both; only `scripts/run_pilot.py` and `scripts/build_dev_grid.py` differ. |
| Frozen analysis configuration (`data/`) | Commit `847cc275`. |
| Exploratory analysis (`results/e189-exploratory-20261022/`) | Commit `b048d3e5` (the private-graph look line, which includes the weighted partition records and gated store sides). |
| Parallel analysis line | Commit `5837f28e` (entry-150 route admission and set-aside exclusion), as a patch from the common base `a538ba9d` of both lines (it does not apply on top of the shipped analysis, which has moved on from that base). The looks over entry-150 retry runs used this line; the looks over the private graph used the line shipped above. |

Each `replication/versions/run-tags/<tag>.patch` turns the shipped framework into the framework of that tag
(`git apply` from the repository root). The tags are listed below.

| Tag | Commit | Used for |
|---|---|---|
| `test-freeze-20261022-sci-c30257bba485` | `126a111b` | The test-split freeze (change-log entries 55, 128). |
| `test-e167-20261022-sci-c30257bba485` | `87aa4bc3` | Entry 167 (run admission) on top of the frozen code. |
| `test-e180-20261022-sci-c30257bba485` | `e203098d` | Shipped as is. Entry 180: per-model request time-out and turn cap. |
| `test-e188-20261022-sci-c30257bba485` | `45c67682` | The probe runner's per-model request time-out (entry 188). |
| `test-e189-20261024-sci-c30257bba485` | `547e129e` | The closed-book test probes (entries 183, 188). |
| `baselines-20261024-ab7da3ad` | `ab7da3ad` | The retry-on-empty and LLM-judge baseline cells. |
| `baselines-sparqlllm-20261024-71f8f2e2` | `71f8f2e2` | The SPARQL-LLM baseline cells. |
| `claude32k-20261024-1735e82a` | `1735e82a` | Raised per-call output cap, new runs only (cap tag A). |
| `cap81920-qwen-deepseek-20261024-34817307` | `34817307` | Raised per-call output cap, new runs only (cap tag B). |
| `cap81920-glm-coder-20261024-12c0b79d` | `12c0b79d` | Raised per-call output cap, new runs only (cap tag C). |
| `claudemax-20261024-8bc0b35a` | `8bc0b35a` | Raised Claude output cap and request time-out, new runs only (cap tag A2). |

Each run records the code version it ran on in its manifest; that manifest is part of the data of step 2.

## Environment

- Python 3.12 (the analysis and the runs used CPython 3.12.3 on x86_64).
- Dependencies are pinned in `uv.lock`: `uv sync --frozen` creates the environment.
- The confirmatory analysis runs single-threaded: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1` (the analysis refuses to
  start otherwise). Its recorded environment is numpy 2.5.4 with OpenBLAS 0.3.34.

## Running

Tests that need no data:

    replication/run_tests.sh

The analysis needs the data of step 2. `replication/reproduce.sh` lists the commands and their inputs; until the data
is released it prints them and exits.

## Anonymization

- People, institutions, computing sites, hosts, services and paths are replaced consistently: computing sites become
  `SiteA`, `SiteB`, `SiteC`; hosts become names under `.example`; the hosted model gateway becomes `gateway`;
  internal addresses become documentation addresses (`192.0.2.0/24`); home directories become `/home/user`; store
  copies whose names carried a port carry a neutral index (`c01` ...). Hashes of store endpoint addresses are the
  hashes of the anonymized addresses.
- Where such a name was part of an identifier, the identifier was renamed the same way in every file (for example,
  `harness/sources/generated_gateway.py`, `scripts/run_gateway_ab.py`, and the options, environment variables and
  constants named after the gateway service). Nothing else in the code changed: for every Python file, the syntax tree
  with all string constants blanked is identical before and after anonymization, except for these consistent renames.
- The analysis pins the digests of the code and configuration it runs on. Anonymization changes file contents, so the
  affected pins were recomputed on the anonymized files; `replication/PINS.md` lists the original and the package
  value of each. The digests recorded inside the data of step 2 refer to the code before anonymization.
- The store-copy tables of the exploratory analysis describe the authors' infrastructure in anonymized form. Their
  free-text fields (bases, notes, citations) read `operational note withheld`, and the digests of the private
  evidence files they cite are zeros; the fields the code reads (copy names, classes, graphs, hosts, endpoint
  hashes, the digests of the store checks) are kept. A replicator who reruns the experiments supplies tables for their
  own store copies (see `replication/reproduce.sh`).
- References in comments to the project's internal design log and review process are neutralized ("design log",
  "design ruling", "study lead", "external reviewer").

## Not included, and why

- **Held-out test data.** The gold answers of the test split are withheld so that the benchmark stays usable and
  uncontaminated. The scripts that build the held-out sets are not included either.
- **The private graph (`a15`).** The store is private and restricted. Its contents, the questions and gold answers
  generated over it, and every output derived from it are not released. The generic code that handles it is
  included; the graph-specific vocabulary and templates live outside the code by design.
- **SIB gold.** Restricted; not released.
- **Operational material.** Operator notes, run logs, job scripts for the computing sites, and probe outputs.
- **Some tests.** The framework's own test suite is not in this version (its fixtures are being reviewed for
  release); the tests of the site job scripts are left out with those scripts, and the analysis tests that need data or the
  development repository's history are listed in `replication/deselected-tests.txt`.

## Data

The recorded run outputs and the analysis inputs follow in step 2, licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## License

Code: MIT (see `LICENSE`). Data (step 2): CC BY 4.0.

## Citation

See `CITATION.cff`.
