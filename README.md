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

## Setup

Use Python 3.11 or newer in a clean environment:

```bash
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
To regenerate all outputs after placing the inputs described in
`data/README.md`, run:

```bash
make evaluate
```

The same command is available as `margins-maps`. The workflow writes CSV files
to `results/`, diagnostic PDFs to `plots/`, and a concise `results/summary.json`.
Generated workloads use fixed random seeds, so non-timing outputs are
deterministic. Runtime measurements naturally depend on the host machine.
