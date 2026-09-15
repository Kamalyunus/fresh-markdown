"""ops.run -- how a driver runs one pipeline module, and the phase order.

`step` is the subprocess runner `ops.bootstrap_loop` and `ops.advance`
both drive the chain through (`python3 -m package.module ...`, output
streamed, a non-zero exit stopping the driver); `PHASES` is the order
`ops.advance` walks and the readiness report renders. Neither knows what
any step does.
"""

import subprocess
import sys

from common.paths import PREPARED                                   # noqa: F401

PHASES = ("data", "bootstrap", "tune", "posterior", "shadow", "owner",
          "launch", "daily")


def step(label, args, fatal=True):
    """One pipeline step. Output streams through: the console lines are the
    evidence, and swallowing them to keep the log tidy is how a warning gets
    missed."""
    print(f"\n== {label} " + "=" * max(0, 62 - len(label)))
    r = subprocess.run([sys.executable, "-m", *args])
    if r.returncode and fatal:
        raise SystemExit(f"\n{label} FAILED (exit {r.returncode}) -- stopping "
                         "here rather than building on a broken artifact")
    return r.returncode
