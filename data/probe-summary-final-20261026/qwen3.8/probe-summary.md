# Closed-book probes (E1.7)

Exact match over attempts (wrong: unparseable, query errors, provider failures, refusals, context caps); 95% intervals by a bootstrap over base questions, repetitions averaged first.

## corporate

### Probe (a): answer from memory

| model | prompt | N | questions | exact | 95% CI | F1 | declined | unparseable | provider fail | refusal | truncated | infra (excluded) |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen/Qwen3.8-27B | closed-book-answer-v2 | 630 | 210 | 0.0% | [0.0, 0.0] | 0.3% | 21.4% | 0.3% | 0.0% | 0.0% | 0.0% | 0 |

### Probe (b): query from memory, executed

| model | N | questions | exact | 95% CI | F1 | parse | refused | store error | time-out | provider fail | infra (excluded) | query from |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| Qwen/Qwen3.8-27B | 630 | 210 | 26.7% | [22.7, 30.6] | 28.1% | 30.2% | 0.0% | 0.3% | 1.7% | 0.0% | 0 | none: 16, reply: 486, sparql_fence: 128 |

## opaque-rhea-142

### Probe (a): answer from memory

| model | prompt | N | questions | exact | 95% CI | F1 | declined | unparseable | provider fail | refusal | truncated | infra (excluded) |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen/Qwen3.8-27B | closed-book-answer-v2 | 630 | 210 | 1.1% | [0.2, 2.5] | 1.1% | 33.2% | 1.7% | 0.0% | 0.0% | 1.1% | 0 |

### Probe (b): query from memory, executed

| model | N | questions | exact | 95% CI | F1 | parse | refused | store error | time-out | provider fail | infra (excluded) | query from |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| Qwen/Qwen3.8-27B | 630 | 210 | 0.0% | [0.0, 0.0] | 2.3% | 47.5% | 0.0% | 1.0% | 29.7% | 0.0% | 0 | none: 17, reply: 343, sparql_fence: 270 |

## rhea-142

### Probe (a): answer from memory

| model | prompt | N | questions | exact | 95% CI | F1 | declined | unparseable | provider fail | refusal | truncated | infra (excluded) |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen/Qwen3.8-27B | closed-book-answer-v2 | 630 | 210 | 4.6% | [2.2, 7.3] | 4.6% | 6.2% | 2.7% | 0.0% | 0.0% | 2.5% | 0 |

### Probe (b): query from memory, executed

| model | N | questions | exact | 95% CI | F1 | parse | refused | store error | time-out | provider fail | infra (excluded) | query from |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| Qwen/Qwen3.8-27B | 630 | 210 | 4.9% | [3.2, 7.0] | 8.7% | 37.8% | 0.5% | 1.4% | 20.2% | 0.0% | 0 | none: 6, reply: 389, sparql_fence: 235 |

## uniprot-sp-2026_03

### Probe (a): answer from memory

| model | prompt | N | questions | exact | 95% CI | F1 | declined | unparseable | provider fail | refusal | truncated | infra (excluded) |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen/Qwen3.8-27B | closed-book-answer-v2 | 630 | 210 | 0.8% | [0.2, 1.7] | 0.8% | 14.8% | 0.2% | 0.0% | 0.0% | 0.2% | 0 |

### Probe (b): query from memory, executed

| model | N | questions | exact | 95% CI | F1 | parse | refused | store error | time-out | provider fail | infra (excluded) | query from |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| Qwen/Qwen3.8-27B | 630 | 210 | 0.0% | [0.0, 0.0] | 0.0% | 40.8% | 0.3% | 1.3% | 31.0% | 0.0% | 0 | none: 11, reply: 406, sparql_fence: 213 |

## Natural minus opaque, probe (b)

Placeholder until both graphs of a pair are run: the difference of exact match (natural minus opaque) per model, beside the agent's opaque-minus-natural difference once the main runs exist.

| natural | opaque | model | natural exact | opaque exact | natural − opaque | agent opaque − natural |
|---|---|---|---:|---:|---:|---:|
| rhea-142 | opaque-rhea-142 | Qwen/Qwen3.8-27B | 4.9% | 0.0% | +4.9 pt | pending (main runs) |
| uniprot-sp-2026_03 | opaque-uniprot-sp-2026_03 | Qwen/Qwen3.8-27B | 0.0% | — | pending | pending (main runs) |
