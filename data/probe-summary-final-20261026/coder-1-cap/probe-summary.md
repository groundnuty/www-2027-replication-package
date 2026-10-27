# Closed-book probes (E1.7)

Exact match over attempts (wrong: unparseable, query errors, provider failures, refusals, context caps); 95% intervals by a bootstrap over base questions, repetitions averaged first.

## opaque-rhea-142

### Probe (a): answer from memory

| model | prompt | N | questions | exact | 95% CI | F1 | declined | unparseable | provider fail | refusal | truncated | infra (excluded) |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen/Qwen3-Coder-30B-A3B-Instruct | closed-book-answer-v2 | 8 | 7 | 0.0% | [0.0, 0.0] | 0.0% | 12.5% | 0.0% | 0.0% | 0.0% | 0.0% | 0 |

### Probe (b): query from memory, executed

| model | N | questions | exact | 95% CI | F1 | parse | refused | store error | time-out | provider fail | infra (excluded) | query from |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| Qwen/Qwen3-Coder-30B-A3B-Instruct | 2 | 2 | 0.0% | [0.0, 0.0] | 0.0% | 50.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0 | reply: 2 |

## rhea-142

### Probe (a): answer from memory

| model | prompt | N | questions | exact | 95% CI | F1 | declined | unparseable | provider fail | refusal | truncated | infra (excluded) |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen/Qwen3-Coder-30B-A3B-Instruct | closed-book-answer-v2 | 3 | 3 | 0.0% | [0.0, 0.0] | 0.0% | 33.3% | 33.3% | 0.0% | 0.0% | 33.3% | 0 |

### Probe (b): query from memory, executed

| model | N | questions | exact | 95% CI | F1 | parse | refused | store error | time-out | provider fail | infra (excluded) | query from |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| Qwen/Qwen3-Coder-30B-A3B-Instruct | 10 | 8 | 0.0% | [0.0, 0.0] | 0.0% | 30.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0 | reply: 3, sparql_fence: 7 |

## uniprot-sp-2026_03

### Probe (a): answer from memory

| model | prompt | N | questions | exact | 95% CI | F1 | declined | unparseable | provider fail | refusal | truncated | infra (excluded) |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen/Qwen3-Coder-30B-A3B-Instruct | closed-book-answer-v2 | 7 | 7 | 0.0% | [0.0, 0.0] | 0.0% | 14.3% | 0.0% | 0.0% | 0.0% | 0.0% | 0 |

### Probe (b): query from memory, executed

| model | N | questions | exact | 95% CI | F1 | parse | refused | store error | time-out | provider fail | infra (excluded) | query from |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| Qwen/Qwen3-Coder-30B-A3B-Instruct | 2 | 2 | 0.0% | [0.0, 0.0] | 0.0% | 50.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0 | reply: 2 |

## Natural minus opaque, probe (b)

Placeholder until both graphs of a pair are run: the difference of exact match (natural minus opaque) per model, beside the agent's opaque-minus-natural difference once the main runs exist.

| natural | opaque | model | natural exact | opaque exact | natural − opaque | agent opaque − natural |
|---|---|---|---:|---:|---:|---:|
| rhea-142 | opaque-rhea-142 | Qwen/Qwen3-Coder-30B-A3B-Instruct | 0.0% | 0.0% | +0.0 pt | pending (main runs) |
| uniprot-sp-2026_03 | opaque-uniprot-sp-2026_03 | Qwen/Qwen3-Coder-30B-A3B-Instruct | 0.0% | — | pending | pending (main runs) |
