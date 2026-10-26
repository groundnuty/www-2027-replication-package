# Known issues

## 1. The checker's offline replay is not deterministic for a few records

**What happens.** When the checker builds the queries that test an answer's footprint, it sometimes picks a
different term or variable name for the same query from one process to the next. The choice happens in
`harness/checks/footprint.py:257` (reached from `:507` and `:305`, called by `harness/checks/compiler.py:468`). The
likely cause is that some intermediate collections are keyed by object identity, whose order depends on memory
addresses and therefore on the process.

**Why it matters.** The replay check (`scripts/verify_batch.py`) recomputes every recorded checker decision offline
and compares it with the decision the online run stored. It looks up the stored store responses by query text. If
the offline run builds one query with a different term, that query has no stored response. The recomputed decision
can then differ from the stored one, and the record is reported as not reproducing ("held"). A failed recompute
writes nothing. A passing recompute reproduces the record's own stored decision, and the stored decisions are the
ones the analysis reports.

**How often we saw it.**

- One record of the runs on the human-written (SIB) UniProt questions failed to reproduce in a share of fresh offline runs (12 diagnostic runs of
  that one record: 8 reproduced, 4 did not). In the one failing run that was diagnosed, every missing query was a
  single query text asked several times: the same length as the nearest stored query, with one or two tokens
  different. It was not a reordering of clauses.
- In one of the further UniProt retry cells (2,502 records), each full replay check held a different small set of records (6, 7,
  4 and 7 in four consecutive runs; under 1% of the cell, one model).
- The other cells' replay checks reproduced every record.

**How the analyses handled it.** The code was not changed: the stores the runs used are no longer live, so a fix
could not be validated against them.

- *Record-level reproduction retries.* A held record was re-checked alone, in a fresh process, at the same seed, up
  to 8 times. A record that reproduced in at least one try was kept, and every try was logged. A record that failed
  all 8 tries was set aside with the frozen `scripts/relabel_harness_failure.py --diverged` as `failed:harness`
  (checker divergence), awaiting a checker fix. That happened to exactly one record (one retry of one model).
  Its parent attempt was then excluded from every arm of that cell's analysis, so the arms stay on one population.
  The exclusion is disclosed in the analysis output (`excluded_setaside_parents`).
- *In-job retries for the partitioned SIB UniProt run.* Each offline pass of the partitioned run's replay was re-run at the
  same seed, up to 8 times, only when the replay itself reported that it did not reproduce the checks. A failed try
  writes nothing, so a retry can only select reproduction success, never an outcome. Acceptance was unchanged: the
  two offline passes had to be byte-identical, and the replayed partition's matrix and store log had to equal the
  ones staged before the run.
- Every retry, held set and set-aside is listed in the run records released with the data.
