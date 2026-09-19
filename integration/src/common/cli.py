"""common.cli -- the argument parser every `python3 -m package.module` shares.

One factory for the flags all the modules spell the same way: `--config`
(default config.yaml), and on request `--reports` (the reports directory)
and `--out` (the module's report path). A module adds its own flags to
the parser it gets back; the defaults live in common.paths.
"""

import argparse

from common import paths


def make_parser(prog=None, reports=False, out=None, **kw):
    """An ArgumentParser carrying `--config`; `reports=True` adds
    `--reports` (default common.paths.REPORTS), `out=<path>` adds `--out`
    with that default. `kw` passes through (description, formatter_class)."""
    ap = argparse.ArgumentParser(prog=prog, **kw)
    if out is not None:
        ap.add_argument("--out", default=out)
    ap.add_argument("--config", default="config.yaml")
    if reports:
        ap.add_argument("--reports", default=paths.REPORTS)
    return ap
