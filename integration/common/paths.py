"""common.paths -- the files the drivers name outside config.yaml.

The extract, the prepared frame, the reports the chain writes by default
and the driver's own journal, decision log and readiness page: each
spelt once here, so a driver's argv (`ops.advance`, `ops.bootstrap_loop`)
and the module's own `--out` default cannot drift apart. Artifacts the
config names (`posterior.path`, `dispersion.r_lookup_path`, ...) stay in
config.yaml -- these are the paths nothing tunes.
"""

RAW = "data/flc_raw.parquet"                 # step 0, fit.download_flc
PREPARED = "data/prepared.parquet"           # step 1, fit.prepare_data

REPORTS = "reports"
BACKTEST_REPORT = "reports/backtest.json"
SHADOW_REPORT = "reports/shadow.json"
MONITOR_REPORT = "reports/monitor.json"
ASSURANCE_REPORT = "reports/assurance.json"
READINESS = "launch_readiness.md"            # under the reports directory

JOURNAL = "artifacts/advance_journal.json"   # what advance ran, per round
DECISIONS = "artifacts/config_decisions.json"  # what tune --apply pasted, and why

EXPORTS = "exports"                          # daily.export_events --out-dir
