# WWW 2027 replication package: data

This directory holds the analysis outputs of the paper's experiments and the tools that recompute from them. It is anonymized
for double-blind review in the same way as the code package: names of people, institutions, computing resources and services are
replaced by neutral labels. License: CC BY 4.0.

The infrastructure sections of the results files are reduced to the model configurations of replication/MODELS.md (A, B, A+B) and the store speed class names; nothing else about how the models or stores were served is released.

## Layout

| Location | What it is |
|---|---|
| `data/<package>/` | One directory per released analysis package (table below): the analysis outputs (`results.json`, tables, macro files), checker ledgers and, where the analysis was rerun for a sensitivity, its call output. The per-attempt tables, the checker ledgers, the replay matrices, the judge calls and the audit population, the macro files (`numbers.tex`, `numbers.json`), the bank plan and run files and the layer-ablation result are gzipped (`.gz`); every other file, `results.json` included, is plain whatever its size (the largest, 7.5 MiB, is a results file). |
| `data/attempts/` | Per-attempt tables, one per run directory the analyses cite (public graphs only); `INDEX.tsv` lists them. |
| `tools/<name>/` | Tools that recompute from the released files, and the table generators. |
| `STAGE-FILES.tsv`, `SHA256SUMS` | Every file with its size and sha256. |

Each released json file keeps only the keys named in a per-file keep list; every other key is dropped.

## Reproducing numbers from the released files

- `numbers.tex` and `numbers.json` of the confirmatory call are regenerated from its `results.json` by `harness.analysis.macros.generate` (code package). `tier1-confirmatory-20261023/results.json` is the registered results file of the pre-registered analysis.
- `tier1-confirmatory-20261023/numbers-part1.tex` is a subset of `numbers.tex` (3,524 macros: the real-output ledger counts, the confirmatory conclusions and the UniProt macros); every line is also a line of `numbers.tex`.
- `exploratory.py compare` (code package) compares two released `results.json` files section by section (arms, real-output ledger, refusals, scorer changes, model settings, output-cap hits, confirmatory).
- `tools/p1p5` reruns P1 to P5 from the dev inputs in `data/p1p5-dev-20261026/` (dev split only). The frozen-source comparison of P1-P5 needs the unreleased source files, so it is not checked: that one reconciliation line reads `NOT CHECKED (sources not released)` and is not counted as a failure; every other reconciliation is checked.
- `tools/table-generators` render the paper's tables from the released results files (the sections on stores and serving are not included).
- Recomputations that need the withheld run records: `tools/cost-h3` (H3 cost), `tools/rhea-gap`, `tools/h6-within` (the H6 helper `h6_analysis.py` is included), the two sensitivity calls (`exclusion-sensitivity-call`, `timeout-sensitivity-call`: they rerun the registered analysis), the fault tables, trivial checkers and layer ablation (they need the frozen fault bank's items), and the baseline-cell estimands and detection wrappers. Each is released as its output (and its tool where it is included); the macros, `compare` and P1 to P5 are the recomputations that run from the released files.
- The builder of the aggregated store time-out table is not included (its output, the speed-class table, is).
- `tools/rhea-gap` and `tools/cost-h3` are shipped with neutral wording: comments, docstrings and messages are reworded and a few names are renamed (the option `--analysis-dir`, the key `configurations_on_both_graphs`, the record names `index.json`, `pilot-grid-index/1` and `build_info`, the helper names `analysis_dir` and `not_configurations`); the syntax trees equal the originals' with every string constant blanked and these names applied, and their test suites run unchanged (27 and 35 tests, all passing).
- A rerun of the shipped `tools/rhea-gap` also writes what the released `table.json` omits: the per-arm configuration shares (the key `shares`), a longer `what` text that mentions them, the key `configurations_on_both_graphs` of each pair, the bootstrap's cluster map and the hashes of its resample matrix and configuration, and the `inputs` and `code` blocks.
- **The confirmatory and exploratory reruns need the withheld test records.** The held-out test gold and the run records (transcripts, queries, answers) are not released, so `scripts/analyse.py`, `scripts/verify_batch.py` and `exploratory.py analyse` cannot be rerun from this directory; the released `results.json` files are the record of those analyses.

## What the released files do not contain, and why

- Test-split gold answers and questions with gold: they would contaminate the benchmark.
- Run records (transcripts, tool-call arguments and results, final messages, refusal and error text): they hold queries and answers.
- The private graph, and every output derived from it: restricted store. Voided-run lists are not released.
- SIB gold and the SIB fact packages: restricted. Of the SIB analysis outputs only the reported runs and their companions are released, with the gold-derived keys (`gold_types_all_questions_run`, `analysed_by_gold_type`, `by_gold_run_status`, `catalogue_examples`, `gold_sources`) removed.
- Replay matrices of test outputs, the frozen fault bank's items and the pinned question files: their check explanations and queries quote gold-derived text.
- Run logs, scripts, identifiers and operator notes: operational detail, not analysis data.
- Annotator sheets, the sealed key and audit items: the items show gold queries and answers (the audit tool ships with a synthetic example only).

## Per-attempt tables and ledgers (allowlist)

- Per-attempt tables (`data/attempts/*.jsonl.gz`, first line a header with the column names) hold only: graph, model, arm, question id, base question id, attempt (repetition) and run id, template, split, status, missing reason, correct, input and output tokens, cost, wall time, tool-call count, the output-cap flag, the model configuration of replication/MODELS.md, written `<model> A`, `<model> B` or `<model> A+B` (for example `Gemma A+B`; `Claude` for the Claude models), given per attempt, with `A+B` only where the attempt's own turns ran under both configurations) and the store speed class (`fast`, `medium`, `slow`, or `fast (assigned, not measured)` and `slow (assigned, not measured)` where a class was assigned rather than measured; `null` where it is missing, 1,946 of 58,199 rows). They do not hold f1, gold-empty flags, alignment, result rows, evidence-triple counts, v4 verdicts, or any query, answer, transcript or feedback text. `tier` and `engine_sensitive` are not recorded in the run records and are therefore not in the tables.
- The tables list the initial attempts of the prompt arms. Retry outcomes and per-source verdicts are in the checker ledgers, which share the run id; a run directory whose records another directory already holds (a union copy) adds no rows. There are no per-attempt tables for the SIB analyses, and none for runs on the private graph.
- Checker ledgers (`ledger.jsonl.gz`, one row per output and checker source) keep the checker-ledger columns without the alignment column (`checked_query`), `result_rows`, `evidence_triples` and `graph_outside_census`; they contain flags, kinds and counts only (no reasons).

## Notes on hashes and identifiers

- Entries for runs on the private graph were removed, so the hashes recorded in the results files refer to the unreleased originals; identifiers are relabelled as in the code package. The digests recorded inside the released files (code digest, input digests) likewise refer to the unreleased originals.
- In the results files the one `restricted` population is labelled `reference deployment`.
- Speed classes are kept by name: `fast`, `medium`, `slow`, with `fast (assigned, not measured)` and `slow (assigned, not measured)` where a class was assigned rather than measured; a missing class is written `null` (1,946 rows of the per-attempt tables).

## Share rows in the results files

The `shares` of the infrastructure section count, per arm, its units by model configuration (replication/MODELS.md): `A` or `B` (all the unit's own turns ran in that configuration), `A+B` (they ran in both) and `Claude` (the Claude models have one configuration). The class shares count the units by the speed class of the store copies they used: `fast`, `medium`, `slow`; `fast (assigned, not measured)` and `slow (assigned, not measured)` where the class was assigned and not measured; `more than one class` where a unit's copies have different classes (283 units in 12 arms); `no class recorded` where a unit's copy has no class in the table (1,079 units in 14 arms). Each block also carries `attempts` (the units with a run record, the denominator of its rows) and `without record` (the units without one: 0 in every block).

## Released packages

| Directory | What it is |
|---|---|
| `data/tier1-confirmatory-20261023` | The confirmatory call on UniProt Tier 1: results.json (also the registered results of the preregistered analysis), numbers.json and numbers.tex (one macro per reported number; regenerated from results.json by harness.analysis.macros.generate). |
| `data/t1-faulttable-v3-20261024T072520Z` | Fault-table estimates per source and column on the frozen UniProt fault bank (ids, classes, rates). |
| `data/t1-trivial-20261023T223729Z` | Trivial checkers (random flagger, retry on empty) on UniProt Tier 1, with the retry-on-empty checker ledger. |
| `data/look-20261023T222012Z` | UniProt per-model checks (three cells): results.json and checker ledger per cell. |
| `data/retry-cells-uniprot-20261026` | Further UniProt retry cells (eleven looks): results.json and checker ledger per cell. |
| `data/retry-cell-reproduction-gemma` | Reproduction of one retry cell. |
| `data/look-20261025T134600Z` | Look: UniProt retry cells with offline checker replays. |
| `data/look-20261025T181300Z` | Look: opaque Rhea cells with offline checker replays. |
| `data/look-20261025T184400Z` | Look: Rhea cells with offline checker replays. |
| `data/look-20261025T185600Z` | Look: corporate-graph cells with offline checker replays. |
| `data/look-20261025T191656Z` | Look: opaque Rhea union with offline checker replays. |
| `data/look-20261025T191817Z` | Look: opaque Rhea cell with offline checker replays. |
| `data/look-20261025T192128Z` | Look: opaque Rhea cells with offline checker replays. |
| `data/look-20261025T193413Z` | Look: opaque Rhea Qwen cells with offline checker replays. |
| `data/look-20261025T194629Z` | Look: corporate-graph cell with offline checker replays. |
| `data/look-20261025T195113Z` | Look: Rhea cell with offline checker replays. |
| `data/look-20261026T031613Z` | Look: Rhea cells (natural, prompted, branch). |
| `data/opaque-qwen-pooled-20261025T063802Z` | Qwen opaque Rhea natural versus prompted, pooled and subset. |
| `data/baseline-cells-empty-20261024` | Baseline retry cell estimands, family retry-on-empty. |
| `data/baseline-cells-judge-20261024` | Baseline retry cell estimands, family LLM judge (six models). |
| `data/baseline-cells-judge-glm-20261025` | Baseline retry cell estimands, family LLM judge with the seventh model. |
| `data/baseline-cells-sparqlllm-20261024` | Baseline retry cell estimands, family SPARQL-LLM. |
| `data/detwrap-obqc-20261024` | Ontology-based query checker detection rows on UniProt Tier 1 natural outputs (checker ledger, trivial checkers). |
| `data/detwrap-sparqlllm-20261024` | SPARQL-LLM detection rows on UniProt Tier 1 natural outputs (checker ledger, trivial checkers, parse comparability). |
| `data/judge-llm-uniprot-bank-20261024` | LLM judge converted to checker ledgers (fault bank and UniProt Tier 1), standard run. |
| `data/judge-llm-glm-20261025` | LLM judge converter run for the seventh judge model. |
| `data/judge-strata-20261024` | Empty / non-empty strata of the trivial checkers. |
| `data/judge-strata-glm-20261025` | Strata re-run with the seventh judge model. |
| `data/layer-ablation-20261024` | Layer ablation of the checker on the fault bank. |
| `data/exclusion-sensitivity-counts-20261025` | Counts table for the exclusion sensitivity, seven Tier 1 models. |
| `data/exclusion-sensitivity-call-20261025` | Exclusion sensitivity: the registered analysis rerun with the exclusions. |
| `data/timeout-sensitivity-call-20261024` | Time-out sensitivity for one model: the registered analysis with that model's Tier 1 batch. |
| `data/cap-sensitivity-deepseek-20261024` | Output-cap sensitivity report per partition. |
| `data/h6-analysis-20261025` | H6 two-agent and three-by-three analyses. |
| `data/h6-within-run-20261024` | H6 within a run. |
| `data/probe-summary-26-20261024-settings` | Closed-book probe settings table. |
| `data/probe-summary-final-20261026` | Closed-book probes, final look: per-model reply rates (rates only). |
| `data/cost-h3-20261026T070015Z` | Cost of the checker retries (H3), UniProt Tier 1. |
| `data/rhea-gap-20261026` | Rhea natural versus opaque Rhea accuracy gap, paired cluster bootstrap. |
| `data/store-timeouts-by-arm-and-speed-class-20261025` | Attempts with a store time-out per model, arm and store speed class (infrastructure fields only), opaque Rhea runs of two models. |
| `data/p1p5-dev-20261026` | P1-P5 on the dev checker replay (dev split only; the tool refuses another split). |
| `data/sib-rhea-cap-substituted-20261024` | SIB Rhea, reported run (cap-substituted): analysis output. |
| `data/sib-uniprot-cap-substituted-20261024` | SIB UniProt, reported run (cap-substituted) and its plain-32k companion: analysis outputs. |
| `data/sib-detection-20261025` | SIB model-free detection baselines (random flagger, retry on empty) of the reported runs. |
| `data/qwen-u-union-reviewed-20261026T1329Z` | SIB UniProt pooled partition union (reviewed rerun): primary and companion analysis outputs, and the run ids of the analysed population. |
| `data/sib-rhea-plain-32k-companion-20261024` | SIB Rhea, plain-32k companion run: analysis output. |
| `data/sib-checker-look-20261025` | SIB checker looks of the reported and the plain-32k companion runs: analysis outputs. |

Tools:

| Directory | What it is |
|---|---|
| `tools/audit-analysis` | Audit analysis code with its synthetic example (no real annotator sheets, key or items). |
| `tools/cost-h3` | Cost of the checker retries (H3), with its tests (neutral wording; see the README). |
| `tools/h6-within` | Analysis of H6 within a run. |
| `tools/p1p5` | P1 to P5 on the dev checker replay, with its tests. |
| `tools/rhea-gap` | Paired cluster bootstrap of the Rhea / opaque Rhea accuracy gap, with its tests (neutral wording; see the README). |
| `tools/table-generators` | Scripts that render the paper's result tables from the released results files (public parts only). |

The per-attempt tables (`data/attempts`) back the looks and calls above.
