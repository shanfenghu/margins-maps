import io
import json
import tarfile

import pandas as pd
import pytest

from margins_maps.verification import (
    compare_frame,
    compare_summary,
    inspect_outputs,
    main,
    require_zero_failures,
)


def test_failed_checks_are_fatal():
    with pytest.raises(RuntimeError, match="endpoint: 1 failure"):
        require_zero_failures([{"audit": "endpoint", "failures": 1}])


@pytest.mark.parametrize("value", [None, -1, float("nan"), float("inf"), 0.5])
def test_invalid_failure_counts_are_rejected(value):
    with pytest.raises(RuntimeError):
        require_zero_failures([{"audit": "endpoint", "failures": value}])


def test_empty_checks_are_rejected():
    with pytest.raises(RuntimeError):
        require_zero_failures([])
    require_zero_failures([{"audit": "endpoint", "failures": 0}])


def test_csv_comparison_allows_roundoff_but_not_changed_integer_counts():
    expected = pd.DataFrame({"count": [100000000], "bound": [1.0]})
    actual = expected.copy()
    actual.loc[0, "bound"] += 1e-9
    compare_frame(expected, actual)
    actual.loc[0, "count"] += 1
    with pytest.raises(AssertionError):
        compare_frame(expected, actual)


def test_csv_comparison_rejects_changed_bounds_and_missing_values():
    expected = pd.DataFrame({"bound": [1.0]})
    for value in [1.1, float("nan"), float("inf")]:
        with pytest.raises(AssertionError):
            compare_frame(expected, pd.DataFrame({"bound": [value]}))


def test_runtime_can_vary_but_must_be_positive():
    compare_frame(pd.DataFrame({"ms": [1.0]}), pd.DataFrame({"ms": [100.0]}))
    with pytest.raises(ValueError):
        compare_frame(pd.DataFrame({"ms": [1.0]}), pd.DataFrame({"ms": [0.0]}))


def test_summary_retains_structural_scaling_checks():
    expected = {
        "scaling": {
            "dynamic_program_max_nodes": 10000,
            "dynamic_program_median_ms": 20.0,
        }
    }
    actual = {
        "scaling": {
            "dynamic_program_max_nodes": 10000,
            "dynamic_program_median_ms": 40.0,
        }
    }
    compare_summary(expected, actual)
    actual["scaling"]["dynamic_program_max_nodes"] = 9999
    with pytest.raises(ValueError):
        compare_summary(expected, actual)


def make_reference(tmp_path):
    (tmp_path / "results").mkdir()
    (tmp_path / "plots").mkdir()
    (tmp_path / "reference").mkdir()
    payload = b"query,bound\na,1.0\n"
    (tmp_path / "results" / "bounds.csv").write_bytes(payload)
    (tmp_path / "plots" / "bounds.pdf").write_bytes(b"%PDF-1.4\n")
    manifest = {"format_version": 1, "results": ["bounds.csv"], "plots": ["bounds.pdf"]}
    with tarfile.open(tmp_path / "reference" / "results.tar.gz", "w:gz") as archive:
        for name, data in [
            ("manifest.json", json.dumps(manifest).encode()),
            ("results/bounds.csv", payload),
        ]:
            member = tarfile.TarInfo(name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))


def test_frozen_bundle_detects_missing_and_changed_outputs(tmp_path):
    make_reference(tmp_path)
    assert inspect_outputs(tmp_path) == []
    (tmp_path / "results" / "bounds.csv").write_text("query,bound\na,2.0\n")
    assert any("bounds.csv" in issue for issue in inspect_outputs(tmp_path))
    (tmp_path / "results" / "bounds.csv").unlink()
    assert any("missing" in issue for issue in inspect_outputs(tmp_path))


def test_extra_outputs_and_missing_plots_are_rejected(tmp_path):
    make_reference(tmp_path)
    (tmp_path / "results" / "extra.csv").write_text("value\n1\n")
    assert any("extra" in issue for issue in inspect_outputs(tmp_path))
    (tmp_path / "plots" / "bounds.pdf").unlink()
    assert any("plot coverage" in issue for issue in inspect_outputs(tmp_path))


def test_verification_command_returns_failure_status(tmp_path, monkeypatch):
    make_reference(tmp_path)
    monkeypatch.setattr("sys.argv", ["verify", "--root", str(tmp_path)])
    assert main() == 0
    (tmp_path / "results" / "bounds.csv").write_text("query,bound\na,9.0\n")
    assert main() == 1
