# Closed-book probes (E1.7)

Exact match over attempts (wrong: unparseable, query errors, provider failures, refusals, context caps); 95% intervals by a bootstrap over base questions, repetitions averaged first.

## corporate

### Probe (a): answer from memory

| model | prompt | N | questions | exact | 95% CI | F1 | declined | unparseable | provider fail | refusal | truncated | infra (excluded) |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| claude-haiku-5-5 | closed-book-answer-v2 | 630 | 210 | 0.0% | [0.0, 0.0] | 0.3% | 68.4% | 0.3% | 0.0% | 0.0% | 0.0% | 0 |
| claude-sonnet-5-5 | closed-book-answer-v2 | 630 | 210 | 0.0% | [0.0, 0.0] | 0.2% | 73.5% | 0.0% | 0.0% | 0.0% | 0.0% | 0 |

### Probe (b): query from memory, executed

| model | N | questions | exact | 95% CI | F1 | parse | refused | store error | time-out | provider fail | infra (excluded) | query from |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| claude-haiku-5-5 | 630 | 210 | 0.0% | [0.0, 0.0] | 0.0% | 4.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0 | sparql_fence: 628 |
| claude-sonnet-5-5 | 630 | 210 | 46.3% | [40.2, 52.2] | 48.1% | 4.6% | 0.0% | 0.2% | 0.6% | 0.0% | 0 | sparql_fence: 630 |

## opaque-rhea-142

### Probe (a): answer from memory

| model | prompt | N | questions | exact | 95% CI | F1 | declined | unparseable | provider fail | refusal | truncated | infra (excluded) |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| claude-haiku-5-5 | closed-book-answer-v2 | 630 | 210 | 1.4% | [0.3, 3.0] | 1.4% | 70.8% | 0.3% | 0.0% | 0.0% | 0.0% | 0 |
| claude-sonnet-5-5 | closed-book-answer-v2 | 630 | 210 | 1.6% | [0.3, 3.2] | 1.6% | 75.4% | 0.0% | 0.0% | 0.0% | 0.0% | 0 |

### Probe (b): query from memory, executed

| model | N | questions | exact | 95% CI | F1 | parse | refused | store error | time-out | provider fail | infra (excluded) | query from |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| claude-haiku-5-5 | 630 | 210 | 1.3% | [0.0, 2.9] | 2.7% | 11.6% | 0.0% | 0.0% | 0.0% | 0.0% | 0 | sparql_fence: 630 |
| claude-sonnet-5-5 | 630 | 210 | 1.1% | [0.0, 2.5] | 5.3% | 3.0% | 0.0% | 0.0% | 4.1% | 0.0% | 0 | reply: 1, sparql_fence: 629 |

## rhea-142

### Probe (a): answer from memory

| model | prompt | N | questions | exact | 95% CI | F1 | declined | unparseable | provider fail | refusal | truncated | infra (excluded) |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| claude-haiku-5-5 | closed-book-answer-v2 | 630 | 210 | 4.8% | [2.4, 7.6] | 6.6% | 24.1% | 4.6% | 0.0% | 0.0% | 0.0% | 0 |
| claude-sonnet-5-5 | closed-book-answer-v2 | 630 | 210 | 9.8% | [6.2, 13.8] | 15.8% | 17.5% | 0.0% | 0.0% | 0.0% | 0.0% | 0 |

### Probe (b): query from memory, executed

| model | N | questions | exact | 95% CI | F1 | parse | refused | store error | time-out | provider fail | infra (excluded) | query from |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| claude-haiku-5-5 | 630 | 210 | 24.8% | [20.0, 30.0] | 27.0% | 12.7% | 0.0% | 0.0% | 0.0% | 0.0% | 0 | sparql_fence: 630 |
| claude-sonnet-5-5 | 630 | 210 | 46.0% | [40.5, 51.6] | 49.6% | 2.9% | 0.0% | 0.0% | 0.2% | 0.0% | 0 | sparql_fence: 630 |

## uniprot-sp-2026_03

### Probe (a): answer from memory

| model | prompt | N | questions | exact | 95% CI | F1 | declined | unparseable | provider fail | refusal | truncated | infra (excluded) |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| claude-haiku-5-5 | closed-book-answer-v2 | 630 | 210 | 1.4% | [0.3, 2.9] | 5.5% | 18.7% | 2.5% | 0.0% | 0.0% | 0.0% | 0 |
| claude-sonnet-5-5 | closed-book-answer-v2 | 630 | 210 | 4.9% | [2.2, 7.9] | 13.8% | 21.1% | 0.3% | 0.0% | 0.0% | 0.0% | 0 |

### Probe (b): query from memory, executed

| model | N | questions | exact | 95% CI | F1 | parse | refused | store error | time-out | provider fail | infra (excluded) | query from |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| claude-haiku-5-5 | 630 | 210 | 39.0% | [33.8, 44.3] | 42.1% | 2.9% | 0.0% | 0.0% | 3.7% | 0.0% | 0 | sparql_fence: 630 |
| claude-sonnet-5-5 | 630 | 210 | 53.7% | [48.4, 58.9] | 55.7% | 0.3% | 0.0% | 0.0% | 1.4% | 0.0% | 0 | reply: 1, sparql_fence: 629 |

## Natural minus opaque, probe (b)

Placeholder until both graphs of a pair are run: the difference of exact match (natural minus opaque) per model, beside the agent's opaque-minus-natural difference once the main runs exist.

| natural | opaque | model | natural exact | opaque exact | natural − opaque | agent opaque − natural |
|---|---|---|---:|---:|---:|---:|
| rhea-142 | opaque-rhea-142 | claude-haiku-5-5 | 24.8% | 1.3% | +23.5 pt | pending (main runs) |
| rhea-142 | opaque-rhea-142 | claude-sonnet-5-5 | 46.0% | 1.1% | +44.9 pt | pending (main runs) |
| uniprot-sp-2026_03 | opaque-uniprot-sp-2026_03 | claude-haiku-5-5 | 39.0% | — | pending | pending (main runs) |
| uniprot-sp-2026_03 | opaque-uniprot-sp-2026_03 | claude-sonnet-5-5 | 53.7% | — | pending | pending (main runs) |
