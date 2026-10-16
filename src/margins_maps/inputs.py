"""Validate the immutable inputs used by the evaluation workflow."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(os.environ.get("MARGINS_MAPS_ROOT", Path.cwd())).resolve()
DATA_ROOT = PROJECT_ROOT / "data" / "input"


@dataclass(frozen=True)
class InputSpec:
    """Expected relative path and SHA-256 digest for one input."""

    path: str
    sha256: str


EXPECTED_INPUTS = (
    InputSpec(
        "usgs/ct85co.txt",
        "b70df0da4a41b20f0842927d2aee090a8f69060f249cfeee303d373b31fcbe27",
    ),
    InputSpec(
        "usgs/ct85w8.txt",
        "7434d95af881b1dd266c82c50db6a3fc9e74d0130475a33a8ae03a566792dbc7",
    ),
    InputSpec(
        "usgs/ct_counties_2010.geojson",
        "43d9c1630ae1f6133977a92e88e3d682bc6dadc9b6030b1a693dc7d12b15a6da",
    ),
    InputSpec(
        "usgs/ct_huc8.geojson",
        "395cd1cd86a94440fbf00022e079e004fa6413b43b03ff4bd76c6c81d9b81f1f",
    ),
    InputSpec(
        "usda/counties_ne_ny_2020.geojson.gz",
        "fea3b0d1dfdb2e0887281e227a96b4537ccc2a506df4071d210ca9110368d3ee",
    ),
    InputSpec(
        "usda/huc6_01.geojson.gz",
        "6b2cc9e63bb9ab608d920a125e2d5f23f505c2185e735c43820086e4fffcedd0",
    ),
    InputSpec(
        "usda/county_farms_2022.csv",
        "4481911dbd6316ed84516549210b8fab15bca19ff104cc0d0abd3e6cb4980e8b",
    ),
    InputSpec(
        "usda/huc6_farms_2022.csv",
        "52a3ea86ff574e1af60df7ab0bb652e008c9a32f7b05729732d5df379d5874d8",
    ),
)


def file_digest(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it all at once."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_inputs(
    root: Path = DATA_ROOT,
    specs: Iterable[InputSpec] = EXPECTED_INPUTS,
) -> list[str]:
    """Return actionable diagnostics for missing or altered inputs."""

    diagnostics = []
    for spec in specs:
        path = root / spec.path
        if not path.is_file():
            diagnostics.append(f"missing: {spec.path}")
        elif file_digest(path) != spec.sha256:
            diagnostics.append(f"digest mismatch: {spec.path}")
    return diagnostics


def validate_inputs(root: Path = DATA_ROOT) -> None:
    """Raise a concise error unless every expected input is present and exact."""

    diagnostics = inspect_inputs(root)
    if diagnostics:
        details = "\n  - ".join(diagnostics)
        raise RuntimeError(
            "Input validation failed. See data/README.md for acquisition "
            f"instructions.\n  - {details}"
        )


def main() -> int:
    """Command-line input validation entry point."""

    diagnostics = inspect_inputs()
    if diagnostics:
        print("Input validation failed:")
        for diagnostic in diagnostics:
            print(f"  - {diagnostic}")
        return 1
    print(f"Validated {len(EXPECTED_INPUTS)} inputs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
