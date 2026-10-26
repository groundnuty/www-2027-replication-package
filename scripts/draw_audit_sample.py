#!/usr/bin/env python3
"""Draw the post-run audit sample from a checker ledger (audit-protocol.md, seed 20261028): the rules, the cells and
the private package's separation are harness/analysis/audit_sample.py's.

    uv run python scripts/draw_audit_sample.py --ledger LEDGER.jsonl --split test --records OUT [OUT ...] \\
        --out SAMPLE.json [--private-out PRIVATE.json] [--config data/analysis/prereg-config.json]
    uv run python scripts/draw_audit_sample.py --ledger DEV-LEDGER.jsonl --split dev --dry-run --records DEV-OUT \\
        --out /tmp/sample.json

--records names the run records the ledger was exported from: each kind C item links to the ontology block its own
record shows the agent was given.

Exit 0: the sample is written; 2: refused, and nothing is written.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness.analysis.audit_sample import main

if __name__ == "__main__":
    sys.exit(main())
