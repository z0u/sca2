"""Tests for `./go find`: the static prose extraction, the section cut, glossary expansion, and the ranking."""

from pathlib import Path

import pytest

from tests.conftest import load_script

find = load_script("find")

REPORT = '''# ruff: noqa: B018
# title: Spill scout

import numpy as np

n = 3

"""
# Spill scout

The lead-in, about the scout.
"""

x = np.zeros(n)

f"""
## Where the spill lands (E1)

The spill lands on `lighten` in {n} runs.
"""

"""
The same section goes on in a second string, about separator latches.

## Discussion

Nothing about that word here.
"""
'''

GLOSSARY = """# Glossary

<!-- a comment -->

Alignment
α
:   How far a state points along e₁.

Band {.manual}
:   Our measurement precision.
"""


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    (tmp_path / "docs" / "m2" / "scout").mkdir(parents=True)
    (tmp_path / "docs" / "m2" / "scout" / "report.py").write_text(REPORT)
    (tmp_path / "docs" / "glossary.md").write_text(GLOSSARY)
    (tmp_path / "docs" / "index.md").write_text(
        "# Index\n\n## Iteration 1\n\n- [Spill scout](./m2/scout/report.py)\n\n    Where the spill lands.\n\n    <!-- mini:figures ./m2/scout/report.py -->\n\n- [Other](./m2/other/report.py)\n\n    Unrelated.\n"
    )
    (tmp_path / "todo" / "science").mkdir(parents=True)
    (tmp_path / "todo" / "science" / "latch.md").write_text(
        "---\nstatus: done\ntags: [spill]\n---\n# Why the separator latches\n\nThe body names α once.\n"
    )
    return tmp_path


def test_words_fold_and_stem():
    assert find.words("The Spills on e₁") == ["spill", "e1"]
    assert find.words("glass latches lines annealing") == ["glass", "latch", "line", "anneal"]


def test_report_prose_is_extracted_statically_with_fields_kept(tree: Path):
    passages = list(find.corpus(tree))
    report = [p for p in passages if p.kind == "report"]
    assert [p.heading for p in report] == ["", "Where the spill lands (E1)", "Discussion"]
    assert "{n} runs" in report[1].text
    assert "separator latches" in report[1].text  # a section runs on through the next string
    assert report[0].title == "Spill scout" and report[0].where == "Spill scout"


def test_report_lines_point_at_the_source(tree: Path):
    source = REPORT.splitlines()
    for p in find.corpus(tree):
        if p.kind == "report" and p.heading:
            assert p.heading.split(" › ")[-1] in source[p.line - 1]


def test_index_entries_point_at_their_report(tree: Path):
    index = [p for p in find.corpus(tree) if p.kind == "index"]
    assert [(p.path, p.title, p.text) for p in index] == [
        ("docs/m2/scout/report.py", "Spill scout", "Where the spill lands."),
        ("docs/m2/other/report.py", "Other", "Unrelated."),
    ]


def test_todo_status_and_line(tree: Path):
    (todo,) = [p for p in find.corpus(tree) if p.kind == "todo"]
    assert (todo.status, todo.title, todo.line) == ("done", "Why the separator latches", 5)


def test_glossary_wordings_expand_the_query(tree: Path):
    groups = find.glossary_forms((tree / "docs" / "glossary.md").read_text())
    assert groups == [("Alignment", "α")]
    assert find.expand("alignment spill", groups) == {"alignment": 1.0, "spill": 1.0, "α": find.ALIAS_WEIGHT}


def test_ranking_prefers_rare_words_and_caps_per_file(tree: Path):
    index = find.Index(list(find.corpus(tree)))
    hits = index.search(find.expand("separator latch", []), limit=10, per_file=1)
    assert [p.path for _, p in hits] == ["todo/science/latch.md", "docs/m2/scout/report.py"]


def test_snippet_centres_on_the_query():
    text = "filler " * 100 + "the separator latch is here " + "filler " * 100
    s = find.snippet(text, {"separator": 1.0, "latch": 1.0}, width=80)
    assert "separator latch" in s and s.startswith("…") and s.endswith("…")


def test_cli_filters_and_json(tree: Path, capsys: pytest.CaptureFixture[str]):
    assert find.main(["spill", "--kind", "report", "--json"], root=tree) == 0
    out = capsys.readouterr().out
    assert '"kind": "report"' in out and '"kind": "todo"' not in out
    assert find.main(["zebra"], root=tree) == 1
    assert find.main(["the", "of"], root=tree) == 2
