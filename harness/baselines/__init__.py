"""The baselines of work package E1.4 (design log): the cheap rivals every ontology
check is compared with.

Modules: ``empty`` (retry on empty: a branch cell of the checker loop, triggered by
an empty final query instead of a checker flag) and ``judge`` (the LLM judge: the
agent's own model reads the question, the final query, its result and the answer,
and accepts or rejects it).
"""
