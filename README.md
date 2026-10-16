# Margins Maps

Margins Maps computes certified bounds for spatial queries when public totals
are released on overlapping geographic supports. It reconstructs the overlap
network, checks feasibility, optimises each query endpoint, emits explicit
witness worlds, and independently verifies the resulting certificates.

The complete evaluation covers Connecticut county and hydrologic-unit totals
from the US Geological Survey, together with county and watershed agricultural
totals from the US Department of Agriculture. It also exercises support
perturbations, edge deletions, exhaustive small networks, and scaling workloads.

The central guarantee is simple: every reported interval contains exactly the
query values attainable by nonnegative overlap-cell allocations that reproduce
both released marginal systems. A singleton interval is therefore an exact
answer; a non-singleton interval exposes the unresolved spatial ambiguity.

## Organisation

- `src/margins_maps/` contains the reusable algorithms and evaluation workflow.
- `experiments/` contains the command-line entry point.
- `tests/` contains small deterministic checks.
- `data/input/` contains the compact public inputs needed by the workflow.
- `data/README.md` records their provenance and validation policy.
- `results/` contains machine-readable evaluation outputs.
- `plots/` contains diagnostic visualisations.
- `docs/algorithm.md` explains the optimisation and certification logic.
- `reference/` contains the frozen numerical regression bundle.

## Setup

Use Python 3.11 or newer in a clean environment:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
```

Run commands from the repository root. To invoke an installed command from
elsewhere, set `MARGINS_MAPS_ROOT` to the checkout path.

Confirm that every included input has its expected digest:

```bash
make check-data
```

## Verification

```bash
make test
make audit
```

`make audit` runs the deterministic checks and validates every Python module.
To regenerate and verify all outputs from the included inputs, run:

```bash
make reproduce
```

`make evaluate` (also available as `margins-maps`) regenerates outputs without
comparing them with the frozen reference. Both commands stop with an error if
any independent verification reports failures. The workflow writes CSV files
to `results/`, diagnostic PDFs to `plots/`, and a concise `results/summary.json`.
Generated workloads use fixed random seeds, so non-timing outputs are
deterministic. Runtime measurements naturally depend on the host machine.

## Expected outcomes

The complete run produces 30 CSV outputs, one summary, and four diagnostic
plots. Its water-use cohort has eight counties, 13 hydrologic units, 30 overlap
cells, 93 feasible measures, and 4,092 queries. Of those queries, 2,273 are exact;
the maximum relative width is one third. The agricultural cohort has 252
queries, comprising 173 derived predicates and 79 released-margin controls.
All verification failure counts must be zero.

`make verify-results` checks existing outputs without rerunning the evaluation.
It compares every CSV cell and non-timing summary field with `reference/`,
checks output coverage, and checks that each expected plot has a PDF header.
Integer counts, text, and Boolean values must match exactly. Floating-point
values use relative tolerance `1e-7` and absolute tolerance `1e-9`; missing-value
positions must also match. Positive finite runtime measurements may vary.
Plots should be inspected visually after changing visualisation code.

## Resources and runtime

Use a CPU-only Linux or macOS environment with 4 GiB RAM and 2 GiB free disk
space. No accelerator, proprietary software, account, or access token is
needed. All eight scientific inputs are bundled, totalling about 12 MiB.
Installation requires network access to the Python package index; the full
evaluation then runs offline.

The audited configuration uses Python 3.12 on ARM64 macOS with an Apple M5 CPU
and 24 GiB RAM. Allow ten minutes for a full evaluation, excluding installation;
this is a working budget rather than a benchmark guarantee. The workflow can
run quietly between its initial command and final summary. CI checks Python
3.11 and 3.13. Its manually triggered workflow also runs the full reproduction.
