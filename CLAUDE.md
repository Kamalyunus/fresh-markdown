Read AGENTS.md for the pipeline run order, artifact map, and the hard rules
(frozen-baseline comparisons, gate windows, sign constraints). The authoritative spec is
docs/design.md.

THE DATA IN THIS REPO IS SYNTHETIC. The owner runs the same pipeline on real
historical production data, and their numbers are not these numbers. Anything
a local run prints is evidence about the fixture only — gate values, ratios,
r, rho, prior scores, week counts, PASS/FAIL verdicts. Never report one as a
finding about their extract, and never advise on their data from one: ask them
for the number first. Structural findings do carry over (code paths, leak and
ordering arguments, arithmetic), so say which of the two you are giving.
