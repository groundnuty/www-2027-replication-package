#!/usr/bin/env python3
"""Compare the real bank-only extension's source-free structure with the structure the calibration ran on, and say
whether the calibration must be redone (harness/analysis/structure_check.py; design ruling, design log).

    uv run python scripts/compare_structure.py --bank REAL_BANK_DIR --calibration CALIBRATION_OUT [--split test]

Reads only the bank items' source-free fields and the calibration output's structure.json. Exit 0 when the
calibration stands, 3 when it must be redone at the real structure; writes structure-check.json beside structure.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness.analysis.bank_bounds import load_bank_items
from harness.analysis.structure_check import compare, structure_from_items


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--calibration", type=Path, required=True, help="the calibration output directory (structure.json)")
    ap.add_argument("--split", default="test")
    args = ap.parse_args(argv)
    used = json.loads((args.calibration / "structure.json").read_text(encoding="utf-8"))
    real = structure_from_items(load_bank_items(args.bank / "items.jsonl"), args.split)
    res = {"bank": str(args.bank), "calibration_structure": used["source"], **compare(real, used["questions"])}
    (args.calibration / "structure-check.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print("recalibrate at the real structure: " + ", ".join(res["failed"]) if res["recalibrate"] else
          "the calibration stands: every structure check passes")
    return 3 if res["recalibrate"] else 0


if __name__ == "__main__":
    sys.exit(main())
