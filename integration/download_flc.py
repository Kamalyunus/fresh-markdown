#!/usr/bin/env python3
"""download_flc.py -- every morning, first: the trailing days of the hourly table
pulled from the warehouse into data/flc.parquet, for build_features.py.

REDSHIFT_* credentials come from ~/.env (or --env-file), never from the
config or the code. Runs from this folder whatever the caller's working
directory. `python3 download_flc.py --help` lists the flags.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)

from pricing.extract import main                                          # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
