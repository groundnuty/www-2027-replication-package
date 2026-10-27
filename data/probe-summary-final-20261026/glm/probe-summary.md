# Closed-book probes (E1.7)

Exact match over attempts (wrong: unparseable, query errors, provider failures, refusals, context caps); 95% intervals by a bootstrap over base questions, repetitions averaged first.

## corporate

### Probe (a): answer from memory

| model | prompt | N | questions | exact | 95% CI | F1 | declined | unparseable | provider fail | refusal | truncated | infra (excluded) |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| zai-org/GLM-5.3-Flash | closed-book-answer-v2 | 120 | 40 | 0.0% | [0.0, 0.0] | 0.6% | 39.2% | 19.2% | 0.0% | 0.0% | 18.3% | 0 |

### Probe (b): query from memory, executed

| model | N | questions | exact | 95% CI | F1 | parse | refused | store error | time-out | provider fail | infra (excluded) | query from |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| zai-org/GLM-5.3-Flash | 120 | 40 | 37.5% | [28.3, 46.7] | 38.5% | 30.0% | 0.0% | 0.8% | 0.8% | 0.0% | 0 | none: 29, reply: 23, sparql_fence: 68 |

## opaque-rhea-142

### Probe (a): answer from memory

| model | prompt | N | questions | exact | 95% CI | F1 | declined | unparseable | provider fail | refusal | truncated | infra (excluded) |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| zai-org/GLM-5.3-Flash | closed-book-answer-v2 | 120 | 40 | 1.7% | [0.0, 4.2] | 1.7% | 8.3% | 61.7% | 0.0% | 0.0% | 61.7% | 0 |

### Probe (b): query from memory, executed

| model | N | questions | exact | 95% CI | F1 | parse | refused | store error | time-out | provider fail | infra (excluded) | query from |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| zai-org/GLM-5.3-Flash | 120 | 40 | 0.0% | [0.0, 0.0] | 2.3% | 54.2% | 0.0% | 0.0% | 8.3% | 0.0% | 0 | none: 61, reply: 7, sparql_fence: 52 |

## rhea-142

### Probe (a): answer from memory

| model | prompt | N | questions | exact | 95% CI | F1 | declined | unparseable | provider fail | refusal | truncated | infra (excluded) |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| zai-org/GLM-5.3-Flash | closed-book-answer-v2 | 120 | 40 | 7.5% | [0.8, 15.8] | 8.3% | 0.0% | 29.2% | 0.0% | 0.0% | 29.2% | 0 |

### Probe (b): query from memory, executed

| model | N | questions | exact | 95% CI | F1 | parse | refused | store error | time-out | provider fail | infra (excluded) | query from |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| zai-org/GLM-5.3-Flash | 120 | 40 | 21.7% | [12.5, 31.7] | 25.9% | 28.3% | 0.0% | 0.0% | 4.2% | 0.0% | 0 | none: 30, reply: 15, sparql_fence: 75 |

## uniprot-sp-2026_03

### Probe (a): answer from memory

| model | prompt | N | questions | exact | 95% CI | F1 | declined | unparseable | provider fail | refusal | truncated | infra (excluded) |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| zai-org/GLM-5.3-Flash | closed-book-answer-v2 | 120 | 40 | 1.7% | [0.0, 4.2] | 4.4% | 2.5% | 12.5% | 0.0% | 0.0% | 9.2% | 0 |

### Probe (b): query from memory, executed

| model | N | questions | exact | 95% CI | F1 | parse | refused | store error | time-out | provider fail | infra (excluded) | query from |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| zai-org/GLM-5.3-Flash | 120 | 40 | 26.7% | [16.7, 37.5] | 26.7% | 9.2% | 0.8% | 0.0% | 8.3% | 0.0% | 0 | none: 6, reply: 14, sparql_fence: 100 |

## Natural minus opaque, probe (b)

Placeholder until both graphs of a pair are run: the difference of exact match (natural minus opaque) per model, beside the agent's opaque-minus-natural difference once the main runs exist.

| natural | opaque | model | natural exact | opaque exact | natural − opaque | agent opaque − natural |
|---|---|---|---:|---:|---:|---:|
| rhea-142 | opaque-rhea-142 | zai-org/GLM-5.3-Flash | 21.7% | 0.0% | +21.7 pt | pending (main runs) |
| uniprot-sp-2026_03 | opaque-uniprot-sp-2026_03 | zai-org/GLM-5.3-Flash | 26.7% | — | pending | pending (main runs) |
