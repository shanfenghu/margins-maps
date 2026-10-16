"""Check regenerated numerical outputs against a frozen reference bundle."""

from __future__ import annotations

import argparse
import io
import json
import math
import tarfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .inputs import PROJECT_ROOT


RELATIVE_TOLERANCE = 1e-7
ABSOLUTE_TOLERANCE = 1e-9
TIMING_SUMMARY_KEYS = {
    "dynamic_program_median_ms",
    "generic_solver_median_ms",
    "generic_solver_slowdown",
}


def require_zero_failures(audits: Iterable[dict]) -> None:
    """Reject failed, missing, negative, or non-integer verification counts."""

    records = list(audits)
    if not records:
        raise RuntimeError("Verification failed: no checks were reported")
    problems = []
    for record in records:
        name = record.get("audit", "unnamed check")
        count = record.get("failures")
        try:
            numeric = float(count)
        except (TypeError, ValueError):
            numeric = math.nan
        if not math.isfinite(numeric) or numeric < 0 or not numeric.is_integer():
            problems.append(f"{name}: invalid failure count {count!r}")
        elif numeric != 0:
            problems.append(f"{name}: {int(numeric)} failure(s)")
    if problems:
        raise RuntimeError("Verification failed: " + "; ".join(problems))


def compare_summary(expected, actual, path: str = "summary") -> None:
    """Compare numerical claims, allowing only measured timings to vary."""

    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(f"{path}: fields differ")
        for key, value in expected.items():
            if path == "summary.scaling" and key in TIMING_SUMMARY_KEYS:
                measured = actual[key]
                if not isinstance(measured, (int, float)) or not (
                    math.isfinite(measured) and measured > 0
                ):
                    raise ValueError(f"{path}.{key}: expected a positive finite timing")
            else:
                compare_summary(value, actual[key], f"{path}.{key}")
    elif isinstance(expected, float):
        if not isinstance(actual, (int, float)) or not math.isclose(
            expected,
            actual,
            rel_tol=RELATIVE_TOLERANCE,
            abs_tol=ABSOLUTE_TOLERANCE,
        ):
            raise ValueError(f"{path}: expected {expected!r}, found {actual!r}")
    elif type(expected) is not type(actual) or expected != actual:
        raise ValueError(f"{path}: expected {expected!r}, found {actual!r}")


def compare_frame(expected: pd.DataFrame, actual: pd.DataFrame) -> None:
    """Compare every cell; integer, Boolean, and text fields must match exactly."""

    if list(expected.columns) != list(actual.columns) or expected.shape != actual.shape:
        raise ValueError("row count or columns differ")
    for column in expected:
        if column == "ms":
            values = pd.to_numeric(actual[column], errors="coerce").to_numpy()
            if not (np.isfinite(values) & (values > 0)).all():
                raise ValueError("runtime measurements must be positive and finite")
        elif pd.api.types.is_float_dtype(expected[column]):
            np.testing.assert_allclose(
                actual[column].to_numpy(),
                expected[column].to_numpy(),
                rtol=RELATIVE_TOLERANCE,
                atol=ABSOLUTE_TOLERANCE,
                equal_nan=True,
                err_msg=f"column {column}",
            )
        else:
            pd.testing.assert_series_equal(
                actual[column], expected[column], check_dtype=False, check_exact=True
            )


def inspect_outputs(root: Path = PROJECT_ROOT) -> list[str]:
    """Return diagnostics without extracting or changing any reference files."""

    problems = []
    reference = root / "reference" / "results.tar.gz"
    try:
        with tarfile.open(reference, "r:gz") as archive:
            manifest = json.load(archive.extractfile("manifest.json"))
            if manifest.get("format_version") != 1:
                return ["unsupported reference format"]
            names = manifest["results"]
            if not names or any(Path(name).name != name for name in names):
                return ["invalid reference output names"]
            existing = {
                p.name
                for p in (root / "results").glob("*")
                if p.suffix in {".csv", ".json"}
            }
            if existing != set(names):
                problems.append(
                    "output coverage differs: "
                    f"missing={sorted(set(names) - existing)}, "
                    f"extra={sorted(existing - set(names))}"
                )
            for name in names:
                path = root / "results" / name
                if not path.is_file():
                    continue
                try:
                    stream = archive.extractfile(f"results/{name}")
                    if name.endswith(".csv"):
                        expected = pd.read_csv(io.BytesIO(stream.read()))
                        compare_frame(expected, pd.read_csv(path))
                        if "failures" in expected.columns:
                            require_zero_failures(pd.read_csv(path).to_dict("records"))
                    else:
                        compare_summary(json.load(stream), json.loads(path.read_text()))
                except (AssertionError, ValueError, TypeError, RuntimeError) as error:
                    problems.append(f"{name}: {str(error)[:500]}")
            plots = manifest["plots"]
            if {p.name for p in (root / "plots").glob("*.pdf")} != set(plots):
                problems.append("plot coverage differs from the reference")
            for name in plots:
                if Path(name).name != name:
                    return ["invalid reference plot name"]
                path = root / "plots" / name
                if path.is_file():
                    with path.open("rb") as stream:
                        if stream.read(5) != b"%PDF-":
                            problems.append(f"{name}: missing PDF header")
    except (OSError, tarfile.TarError, KeyError, ValueError, AttributeError) as error:
        problems.append(f"cannot read outputs or reference bundle: {error}")
    return problems


def main() -> int:
    """Validate existing outputs and signal every mismatch with a nonzero exit."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    problems = inspect_outputs(args.root)
    if problems:
        print("Result verification failed:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("All numerical outputs match the frozen reference; plot files are present.")
    print("Measured runtimes may differ; all verification failure counts are zero.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
