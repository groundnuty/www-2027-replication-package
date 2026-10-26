"""Entry 208 exploratory look: run exploratory.py's own main() unchanged, and on failure write ONLY the exception class
and the raising frame's file:line to fd 3 (the operator's channel). main()'s stdout and stderr go wherever the caller
redirected them (the mode-700 look directory, never read by agents).

    python -I look_wrap.py SCRIPT_DIR analyse ARGS...   3>>STATUS 1>OUT/stdout.txt 2>OUT/stderr.txt

run() and load_frozen() are wrapped only to keep the exception object main() catches and prints; nothing else changes."""
import os
import sys
import traceback

script_dir, argv = sys.argv[1], sys.argv[2:]
sys.path.insert(0, script_dir)
import exploratory as X  # noqa: E402

seen = {}


def keep(fn):
    def wrapped(*a, **kw):
        try:
            return fn(*a, **kw)
        except BaseException as e:
            seen["e"] = e
            raise
    return wrapped


X.run = keep(X.run)
X.load_frozen = keep(X.load_frozen)


def diag(rc, e):
    if e is None:
        line = f"rc={rc}"
    else:
        last = traceback.extract_tb(e.__traceback__)[-1]
        line = f"rc={rc} {type(e).__name__} {os.path.basename(last.filename)}:{last.lineno}"
    try:
        os.write(3, (line + "\n").encode())
    except OSError:
        pass


try:
    rc = X.main(argv)
except BaseException as e:  # noqa: BLE001  (not one of the refusals main() catches)
    diag(1, e)
    sys.exit(1)
diag(rc, seen.get("e") if rc else None)
sys.exit(rc)
