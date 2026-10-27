# Closed-book probes (E1.7)

Exact match over attempts (wrong: unparseable, query errors, provider failures, refusals, context caps); 95% intervals by a bootstrap over base questions, repetitions averaged first.

## opaque-rhea-142

### Probe (a): answer from memory

| model | prompt | N | questions | exact | 95% CI | F1 | declined | unparseable | provider fail | refusal | truncated | infra (excluded) |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|

### Probe (b): query from memory, executed

| model | N | questions | exact | 95% CI | F1 | parse | refused | store error | time-out | provider fail | infra (excluded) | query from |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| zai-org/GLM-5.3-Flash | 23 | 23 | 0.0% | [0.0, 0.0] | 0.0% | 39.1% | 0.0% | 0.0% | 13.0% | 0.0% | 0 | none: 8, reply: 1, sparql_fence: 14 |

## rhea-142

### Probe (a): answer from memory

| model | prompt | N | questions | exact | 95% CI | F1 | declined | unparseable | provider fail | refusal | truncated | infra (excluded) |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|

### Probe (b): query from memory, executed

| model | N | questions | exact | 95% CI | F1 | parse | refused | store error | time-out | provider fail | infra (excluded) | query from |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| zai-org/GLM-5.3-Flash | 22 | 18 | 13.9% | [0.0, 30.6] | 13.9% | 22.7% | 0.0% | 0.0% | 0.0% | 0.0% | 0 | none: 5, reply: 2, sparql_fence: 15 |

## uniprot-sp-2026_03

### Probe (a): answer from memory

| model | prompt | N | questions | exact | 95% CI | F1 | declined | unparseable | provider fail | refusal | truncated | infra (excluded) |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|

### Probe (b): query from memory, executed

| model | N | questions | exact | 95% CI | F1 | parse | refused | store error | time-out | provider fail | infra (excluded) | query from |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| zai-org/GLM-5.3-Flash | 2 | 2 | 50.0% | [0.0, 100.0] | 50.0% | 50.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0 | none: 1, sparql_fence: 1 |

## Natural minus opaque, probe (b)

Placeholder until both graphs of a pair are run: the difference of exact match (natural minus opaque) per model, beside the agent's opaque-minus-natural difference once the main runs exist.

| natural | opaque | model | natural exact | opaque exact | natural − opaque | agent opaque − natural |
|---|---|---|---:|---:|---:|---:|
| rhea-142 | opaque-rhea-142 | zai-org/GLM-5.3-Flash | 13.9% | 0.0% | +13.9 pt | pending (main runs) |
| uniprot-sp-2026_03 | opaque-uniprot-sp-2026_03 | zai-org/GLM-5.3-Flash | 50.0% | — | pending | pending (main runs) |
