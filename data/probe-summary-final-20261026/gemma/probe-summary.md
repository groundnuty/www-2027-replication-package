# Closed-book probes (E1.7)

Exact match over attempts (wrong: unparseable, query errors, provider failures, refusals, context caps); 95% intervals by a bootstrap over base questions, repetitions averaged first.

## corporate

### Probe (a): answer from memory

| model | prompt | N | questions | exact | 95% CI | F1 | declined | unparseable | provider fail | refusal | truncated | infra (excluded) |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| google/gemma-4-31B | closed-book-answer-v2 | 630 | 210 | 0.2% | [0.0, 0.5] | 0.2% | 9.7% | 9.4% | 0.0% | 0.0% | 0.0% | 0 |

### Probe (b): query from memory, executed

| model | N | questions | exact | 95% CI | F1 | parse | refused | store error | time-out | provider fail | infra (excluded) | query from |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| google/gemma-4-31B | 630 | 210 | 0.0% | [0.0, 0.0] | 0.0% | 6.2% | 0.0% | 0.0% | 0.0% | 0.0% | 0 | sparql_fence: 630 |

## opaque-rhea-142

### Probe (a): answer from memory

| model | prompt | N | questions | exact | 95% CI | F1 | declined | unparseable | provider fail | refusal | truncated | infra (excluded) |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| google/gemma-4-31B | closed-book-answer-v2 | 630 | 210 | 0.3% | [0.0, 0.8] | 0.3% | 4.3% | 34.6% | 0.0% | 0.0% | 0.6% | 0 |

### Probe (b): query from memory, executed

| model | N | questions | exact | 95% CI | F1 | parse | refused | store error | time-out | provider fail | infra (excluded) | query from |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| google/gemma-4-31B | 630 | 210 | 1.0% | [0.0, 2.2] | 1.0% | 50.5% | 0.0% | 0.0% | 0.0% | 0.0% | 0 | sparql_fence: 630 |

## rhea-142

### Probe (a): answer from memory

| model | prompt | N | questions | exact | 95% CI | F1 | declined | unparseable | provider fail | refusal | truncated | infra (excluded) |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| google/gemma-4-31B | closed-book-answer-v2 | 630 | 210 | 2.5% | [0.6, 4.8] | 2.5% | 4.3% | 24.9% | 0.0% | 0.0% | 0.5% | 0 |

### Probe (b): query from memory, executed

| model | N | questions | exact | 95% CI | F1 | parse | refused | store error | time-out | provider fail | infra (excluded) | query from |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| google/gemma-4-31B | 630 | 210 | 0.0% | [0.0, 0.0] | 0.0% | 74.8% | 0.0% | 0.0% | 0.0% | 0.0% | 0 | sparql_fence: 630 |

## uniprot-sp-2026_03

### Probe (a): answer from memory

| model | prompt | N | questions | exact | 95% CI | F1 | declined | unparseable | provider fail | refusal | truncated | infra (excluded) |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| google/gemma-4-31B | closed-book-answer-v2 | 630 | 210 | 0.0% | [0.0, 0.0] | 0.0% | 9.5% | 22.7% | 0.0% | 0.0% | 0.0% | 0 |

### Probe (b): query from memory, executed

| model | N | questions | exact | 95% CI | F1 | parse | refused | store error | time-out | provider fail | infra (excluded) | query from |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| google/gemma-4-31B | 630 | 210 | 0.6% | [0.0, 1.7] | 0.6% | 39.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0 | sparql_fence: 630 |

## Natural minus opaque, probe (b)

Placeholder until both graphs of a pair are run: the difference of exact match (natural minus opaque) per model, beside the agent's opaque-minus-natural difference once the main runs exist.

| natural | opaque | model | natural exact | opaque exact | natural − opaque | agent opaque − natural |
|---|---|---|---:|---:|---:|---:|
| rhea-142 | opaque-rhea-142 | google/gemma-4-31B | 0.0% | 1.0% | -1.0 pt | pending (main runs) |
| uniprot-sp-2026_03 | opaque-uniprot-sp-2026_03 | google/gemma-4-31B | 0.6% | — | pending | pending (main runs) |
