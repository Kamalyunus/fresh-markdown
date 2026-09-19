#!/usr/bin/env python3
"""check_inputs.py -- the snapshot, the feed, the failed pushes and the hour's response, checked.

Part of this folder; the code is src/ops/check_inputs.py. Runs from this
folder whatever the caller's working directory, so a relative path in an
argument is relative to the folder.

    python3 check_inputs.py --help
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, os.path.join(HERE, "src"))

from ops.check_inputs import main                                        # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
