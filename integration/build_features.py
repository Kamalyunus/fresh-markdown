#!/usr/bin/env python3
"""build_features.py -- every morning, after download_flc.py: the day's extract in, the day's feature table out.

Runs from this folder whatever the caller's working directory, so a
relative path in an argument is relative to the folder. The code is
pricing/. `python3 build_features.py --help` lists the flags.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)

from pricing.features import main                                          # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
