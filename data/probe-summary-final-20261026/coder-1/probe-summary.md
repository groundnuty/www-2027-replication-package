# Closed-book probes (E1.7)

Exact match over attempts (wrong: unparseable, query errors, provider failures, refusals, context caps); 95% intervals by a bootstrap over base questions, repetitions averaged first.

## corporate

### Probe (a): answer from memory

| model | prompt | N | questions | exact | 95% CI | F1 | declined | unparseable | provider fail | refusal | truncated | infra (excluded) |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen/Qwen3-Coder-30B-A3B-Instruct | closed-book-answer-v2 | 630 | 210 | 0.0% | [0.0, 0.0] | 0.0% | 47.5% | 1.1% | 0.0% | 0.0% | 0.0% | 0 |

### Probe (b): query from memory, executed

| model | N | questions | exact | 95% CI | F1 | parse | refused | store error | time-out | provider fail | infra (excluded) | query from |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| Qwen/Qwen3-Coder-30B-A3B-Instruct | 630 | 210 | 0.0% | [0.0, 0.0] | 0.0% | 12.7% | 0.3% | 0.0% | 0.0% | 0.0% | 0 | reply: 564, sparql_fence: 66 |

## opaque-rhea-142

### Probe (a): answer from memory

| model | prompt | N | questions | exact | 95% CI | F1 | declined | unparseable | provider fail | refusal | truncated | infra (excluded) |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen/Qwen3-Coder-30B-A3B-Instruct | closed-book-answer-v2 | 630 | 210 | 0.8% | [0.0, 2.1] | 0.8% | 66.0% | 1.3% | 0.0% | 0.0% | 1.3% | 0 |

### Probe (b): query from memory, executed

| model | N | questions | exact | 95% CI | F1 | parse | refused | store error | time-out | provider fail | infra (excluded) | query from |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| Qwen/Qwen3-Coder-30B-A3B-Instruct | 630 | 210 | 1.1% | [0.0, 2.5] | 1.2% | 32.2% | 0.0% | 0.0% | 0.0% | 0.0% | 0 | reply: 219, sparql_fence: 411 |

## rhea-142

### Probe (a): answer from memory

| model | prompt | N | questions | exact | 95% CI | F1 | declined | unparseable | provider fail | refusal | truncated | infra (excluded) |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen/Qwen3-Coder-30B-A3B-Instruct | closed-book-answer-v2 | 630 | 210 | 0.5% | [0.0, 1.4] | 0.7% | 63.0% | 0.8% | 0.0% | 0.0% | 0.5% | 0 |

### Probe (b): query from memory, executed

| model | N | questions | exact | 95% CI | F1 | parse | refused | store error | time-out | provider fail | infra (excluded) | query from |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| Qwen/Qwen3-Coder-30B-A3B-Instruct | 630 | 210 | 0.0% | [0.0, 0.0] | 0.2% | 40.0% | 0.2% | 0.2% | 0.2% | 0.0% | 0 | reply: 221, sparql_fence: 409 |

## uniprot-sp-2026_03

### Probe (a): answer from memory

| model | prompt | N | questions | exact | 95% CI | F1 | declined | unparseable | provider fail | refusal | truncated | infra (excluded) |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen/Qwen3-Coder-30B-A3B-Instruct | closed-book-answer-v2 | 630 | 210 | 0.0% | [0.0, 0.0] | 0.0% | 46.2% | 1.4% | 0.0% | 0.0% | 1.1% | 0 |

### Probe (b): query from memory, executed

| model | N | questions | exact | 95% CI | F1 | parse | refused | store error | time-out | provider fail | infra (excluded) | query from |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| Qwen/Qwen3-Coder-30B-A3B-Instruct | 630 | 210 | 1.0% | [0.3, 1.7] | 1.0% | 27.9% | 1.0% | 0.0% | 0.5% | 0.0% | 0 | reply: 562, sparql_fence: 68 |

## Natural minus opaque, probe (b)

Placeholder until both graphs of a pair are run: the difference of exact match (natural minus opaque) per model, beside the agent's opaque-minus-natural difference once the main runs exist.

| natural | opaque | model | natural exact | opaque exact | natural − opaque | agent opaque − natural |
|---|---|---|---:|---:|---:|---:|
| rhea-142 | opaque-rhea-142 | Qwen/Qwen3-Coder-30B-A3B-Instruct | 0.0% | 1.1% | -1.1 pt | pending (main runs) |
| uniprot-sp-2026_03 | opaque-uniprot-sp-2026_03 | Qwen/Qwen3-Coder-30B-A3B-Instruct | 1.0% | — | pending | pending (main runs) |
