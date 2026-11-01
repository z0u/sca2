"""Tests for the prose-field check — `{…}` fields in literate prose that span lines."""

from pathlib import Path

from tests.conftest import load_script

check = load_script("prose_fields")


def literate(tmp_path: Path, body: str, name: str = "report.py") -> Path:
    """A literate script (`# title:` header) whose cells and prose are *body*."""
    path = tmp_path / name
    path.write_text("# title: T\n\n" + body)
    return path


def test_a_field_that_ruff_expands_is_flagged(tmp_path: Path):
    """The shape `ruff format` leaves behind: the field's call on lines of its own, splitting the paragraph."""
    script = literate(tmp_path, 'f"""\nSee\n\n{\n    table(\n        rows,\n    )\n}\n"""\n')

    (finding,) = check.findings_in(script)
    assert finding.line == 6
    assert str(finding).endswith(":6: a field in prose spans lines, so ruff format splits the paragraph")


def test_one_line_fields_are_fine(tmp_path: Path):
    script = literate(tmp_path, 'TABLE = table(\n    rows,\n)\n\nf"""\nThe value is {x:.2f}.\n\n{TABLE}\n"""\n')

    assert check.findings_in(script) == []


def test_a_plain_module_has_no_prose(tmp_path: Path):
    path = tmp_path / "experiment.py"
    path.write_text('f"""{\n    x\n}"""\n')

    assert check.findings_in(path) == []
