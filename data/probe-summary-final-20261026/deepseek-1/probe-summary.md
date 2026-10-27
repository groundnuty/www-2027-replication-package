# Closed-book probes (E1.7)

Exact match over attempts (wrong: unparseable, query errors, provider failures, refusals, context caps); 95% intervals by a bootstrap over base questions, repetitions averaged first.

## corporate

### Probe (a): answer from memory

| model | prompt | N | questions | exact | 95% CI | F1 | declined | unparseable | provider fail | refusal | truncated | infra (excluded) |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| deepseek-ai/DeepSeek-V4-Flash-0731 | closed-book-answer-v2 | 630 | 210 | 0.0% | [0.0, 0.0] | 0.4% | 48.3% | 3.8% | 0.0% | 0.0% | 0.0% | 0 |

### Probe (b): query from memory, executed

| model | N | questions | exact | 95% CI | F1 | parse | refused | store error | time-out | provider fail | infra (excluded) | query from |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| deepseek-ai/DeepSeek-V4-Flash-0731 | 630 | 210 | 0.0% | [0.0, 0.0] | 0.0% | 15.9% | 0.0% | 0.0% | 0.0% | 0.0% | 0 | plain_fence: 23, reply: 34, sparql_fence: 573 |

## opaque-rhea-142

### Probe (a): answer from memory

| model | prompt | N | questions | exact | 95% CI | F1 | declined | unparseable | provider fail | refusal | truncated | infra (excluded) |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| deepseek-ai/DeepSeek-V4-Flash-0731 | closed-book-answer-v2 | 630 | 210 | 1.1% | [0.2, 2.4] | 1.1% | 69.5% | 3.2% | 0.0% | 0.0% | 0.2% | 0 |

### Probe (b): query from memory, executed

| model | N | questions | exact | 95% CI | F1 | parse | refused | store error | time-out | provider fail | infra (excluded) | query from |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| deepseek-ai/DeepSeek-V4-Flash-0731 | 630 | 210 | 0.6% | [0.0, 1.6] | 1.8% | 39.0% | 0.0% | 0.0% | 0.8% | 0.0% | 0 | plain_fence: 106, reply: 98, sparql_fence: 426 |

## rhea-142

### Probe (a): answer from memory

| model | prompt | N | questions | exact | 95% CI | F1 | declined | unparseable | provider fail | refusal | truncated | infra (excluded) |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| deepseek-ai/DeepSeek-V4-Flash-0731 | closed-book-answer-v2 | 630 | 210 | 2.2% | [0.8, 4.0] | 2.3% | 54.1% | 5.1% | 0.0% | 0.0% | 0.8% | 0 |

### Probe (b): query from memory, executed

| model | N | questions | exact | 95% CI | F1 | parse | refused | store error | time-out | provider fail | infra (excluded) | query from |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| deepseek-ai/DeepSeek-V4-Flash-0731 | 630 | 210 | 1.4% | [0.2, 3.0] | 2.5% | 38.6% | 0.0% | 0.0% | 0.2% | 0.0% | 0 | plain_fence: 116, reply: 110, sparql_fence: 404 |

## uniprot-sp-2026_03

### Probe (a): answer from memory

| model | prompt | N | questions | exact | 95% CI | F1 | declined | unparseable | provider fail | refusal | truncated | infra (excluded) |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| deepseek-ai/DeepSeek-V4-Flash-0731 | closed-book-answer-v2 | 630 | 210 | 0.2% | [0.0, 0.5] | 0.4% | 46.2% | 7.9% | 0.0% | 0.0% | 0.8% | 0 |

### Probe (b): query from memory, executed

| model | N | questions | exact | 95% CI | F1 | parse | refused | store error | time-out | provider fail | infra (excluded) | query from |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| deepseek-ai/DeepSeek-V4-Flash-0731 | 630 | 210 | 1.9% | [0.8, 3.3] | 2.3% | 23.0% | 0.8% | 0.0% | 0.5% | 0.0% | 0 | plain_fence: 78, reply: 20, sparql_fence: 532 |

## Natural minus opaque, probe (b)

Placeholder until both graphs of a pair are run: the difference of exact match (natural minus opaque) per model, beside the agent's opaque-minus-natural difference once the main runs exist.

| natural | opaque | model | natural exact | opaque exact | natural − opaque | agent opaque − natural |
|---|---|---|---:|---:|---:|---:|
| rhea-142 | opaque-rhea-142 | deepseek-ai/DeepSeek-V4-Flash-0731 | 1.4% | 0.6% | +0.8 pt | pending (main runs) |
| uniprot-sp-2026_03 | opaque-uniprot-sp-2026_03 | deepseek-ai/DeepSeek-V4-Flash-0731 | 1.9% | — | pending | pending (main runs) |
