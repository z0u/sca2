"""The tuning cases for `./go find`: real queries against the real corpus, each with the place it should rank in the top five. A change to the ranking that loses a case shows up here."""

import json
from pathlib import Path

import pytest

from tests.conftest import load_script

find = load_script("find")

CASES = json.loads((Path(__file__).parent / "find_cases.json").read_text())["cases"]
TOP = 5


@pytest.fixture(scope="module")
def ranked():
    glossary = find.glossary_forms((find.ROOT / "docs" / "glossary.md").read_text())
    index = find.Index(list(find.corpus(find.ROOT)))
    return lambda query: index.search(find.expand(query, glossary), TOP, 2)


@pytest.mark.parametrize("case", CASES, ids=[c["query"] for c in CASES])
def test_case_ranks_in_the_top_five(ranked, case):
    hits = ranked(case["query"])
    assert find.rank_of(hits, case["path"]), f"{case['path']} not in {[p.path for _, p in hits]}"
