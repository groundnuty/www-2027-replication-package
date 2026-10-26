"""The checker loop (work package E1.2, design log): online checks with one retry
on fixed, source-neutral feedback (or the matched generic message), the union
arm, the offline replay matrix and the per-output accounting.

Modules: ``checked`` (rule 22 from a record's tool calls), ``arms`` (an arm's
verdict: four-way status, fired checks, retry decision), ``feedback`` (the
fixed format and the generic message), ``store_log`` (the store responses the
checks read, for offline replay), ``online`` (the checker inside the agent
loop, decision replay, branching a retry from a recorded initial output),
``matrix`` (the offline replay matrix) and ``accounting`` (six cells and retry
transitions).
"""
