"""Build a SYNTHETIC audit (real builder, fake records) and run audit_analysis.py's CLI on it.
Usage (from the worktree): uv run --offline python <this> OUTDIR   (OUTDIR new, outside the repository)."""
import json
import shutil
import sys
from pathlib import Path

WT = Path("/work/audit-package")
sys.path.insert(0, str(WT / "tests"))
import test_audit_analysis as t  # noqa: E402

from harness.agent import prompts  # noqa: E402
from harness.analysis import audit_analysis as aa  # noqa: E402

out = Path(sys.argv[1]).resolve()
out.mkdir(parents=True, mode=0o700)
work = out / ".work"
blocks = work / "blocks"
blocks.mkdir(parents=True)
prompts.BLOCK_DIR = blocks
prompts._EXTRA_STORES = []
a = t.Audit(work / "audit", blocks)
ex = out / "synthetic-example"
ex.mkdir(mode=0o700)
shutil.copytree(a.pkg, ex / "package")
shutil.copy(a.key, ex / "key.json")
shutil.copy(a.sheets["annotator-1"], ex / "annotator-1-filled.csv")
shutil.copy(a.sheets["annotator-2"], ex / "annotator-2-filled.csv")
shutil.copy(a.adj, ex / "adjudicated.csv")
flags = t.flags_file(a, ex / "flags.csv")
# the real CLI, on the copies
rc = aa.main(["--package", str(ex / "package"), "--sheet", f"annotator-1={ex / 'annotator-1-filled.csv'}",
              "--sheet", f"annotator-2={ex / 'annotator-2-filled.csv'}", "--adjudicated", str(ex / "adjudicated.csv"),
              "--key", str(ex / "key.json"), "--flags", str(flags), "--out", str(ex / "result"), "--print"])
shutil.rmtree(work)
print("rc", rc)
sys.exit(rc)
