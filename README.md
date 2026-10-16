# Margins Maps

Margins Maps computes certified bounds for spatial queries when public totals
are released on overlapping geographic supports. It reconstructs the overlap
network, checks feasibility, optimises each query endpoint, emits explicit
witness worlds, and independently verifies the resulting certificates.

The complete evaluation covers Connecticut county and hydrologic-unit totals
from the US Geological Survey, together with county and watershed agricultural
totals from the US Department of Agriculture. It also exercises support
perturbations, edge deletions, exhaustive small networks, and scaling workloads.

## Organisation

- `src/margins_maps/` contains the reusable algorithms and evaluation workflow.
- `experiments/` contains the command-line entry point.
- `tests/` contains small deterministic checks.
- `data/` documents the required public inputs; downloaded inputs remain local.
- `results/` contains machine-readable evaluation outputs.
- `plots/` contains diagnostic visualisations.

## Setup

Use Python 3.11 or newer in a clean environment:

```bash
python -m pip install -e '.[dev]'
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

The workflow is deterministic and uses fixed random seeds for synthetic checks.
