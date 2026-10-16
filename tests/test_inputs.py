import hashlib

from margins_maps.inputs import InputSpec, inspect_inputs


def test_input_inspection_reports_missing_file(tmp_path):
    specs = [InputSpec("source/value.bin", "unused")]
    assert inspect_inputs(tmp_path, specs) == ["missing: source/value.bin"]


def test_input_inspection_accepts_expected_digest(tmp_path):
    payload = b"immutable scientific input"
    path = tmp_path / "source" / "value.bin"
    path.parent.mkdir()
    path.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()
    assert inspect_inputs(tmp_path, [InputSpec("source/value.bin", expected)]) == []


def test_input_inspection_rejects_changed_content(tmp_path):
    path = tmp_path / "source" / "value.bin"
    path.parent.mkdir()
    path.write_bytes(b"changed")
    diagnostics = inspect_inputs(
        tmp_path,
        [InputSpec("source/value.bin", hashlib.sha256(b"original").hexdigest())],
    )
    assert diagnostics == ["digest mismatch: source/value.bin"]
