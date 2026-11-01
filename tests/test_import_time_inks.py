"""Tests for the import-time ink check — `light_dark` calls that resolve before a themed figure draws."""

from pathlib import Path

from tests.conftest import load_script

check = load_script("import_time_inks")


def module(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "report.py"
    path.write_text(body)
    return path


def test_a_module_level_ink_is_flagged(tmp_path: Path):
    """A dict of inks built at import resolves once, under the light theme."""
    script = module(
        tmp_path, 'INK = {\n    "a": light_dark("#000", "#fff"),\n    "b": light_dark("#111", "#eee"),\n}\n'
    )

    assert [f.line for f in check.findings_in(script)] == [2, 3]


def test_an_ink_resolved_inside_a_function_is_fine(tmp_path: Path):
    """Inside the plot, the call sees the theme the figure is drawing under."""
    script = module(
        tmp_path,
        'INK = {"a": ("#000", "#fff")}\n\ndef draw():\n    return light_dark(*INK["a"])\n\nf = lambda: light_dark(1, 2)\n',
    )

    assert check.findings_in(script) == []


def test_definition_time_parts_of_a_function_count(tmp_path: Path):
    """A default argument is evaluated when the function is defined, which is at import."""
    script = module(tmp_path, 'def draw(color=light_dark("#000", "#fff")):\n    return color\n')

    assert [f.line for f in check.findings_in(script)] == [1]


def test_a_class_body_runs_at_import(tmp_path: Path):
    script = module(tmp_path, 'class Style:\n    ink = vis.light_dark("#000", "#fff")\n')

    assert [f.line for f in check.findings_in(script)] == [2]
