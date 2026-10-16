# Numerical reference

`results.tar.gz` preserves the original numerical outputs that passed an
independent clean-environment rerun. It contains the 30 CSV outputs, the summary,
and a manifest listing those outputs and the four expected plot names. Archive
owner names and timestamps are cleared. It requires neither Git history nor a
network connection, so the same checks work from a downloaded ZIP archive.

The evaluation writes to `results/` and `plots/`; it never updates this bundle.
Run `make verify-results` to compare existing outputs or `make reproduce` to
regenerate and compare them. Every non-timing CSV cell is checked in row order.
Counts, text, and Boolean values must match exactly; floating-point comparisons
use `rtol=1e-7` and `atol=1e-9`, preserving missing-value positions. Runtime
measurements must be positive and finite but need not match the stored timings.
The expected plot names and PDF headers are checked, not their pixels.

Any mismatch or nonzero failure count makes the command fail. A reference
change should receive independent review together with the corresponding
algorithm, data, and expected-outcome changes; refreshing it is not part of
normal evaluation.
