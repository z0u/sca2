"""Tests for ``mini.lit``: parsing, weaving, incremental re-runs, the memo, and the page."""

import os
import sys
import textwrap
from pathlib import Path

import pytest

from mini.lit import LazyNpz, Runner, is_literate_script, memo, parse, read_npz, render
from mini.lit import set_cache_dir
from mini.lit.document import Cell, Prose
from mini.lit.page import page, to_html
from mini.reports import Publisher


def write(tmp_path: Path, text: str, name: str = "doc.py") -> Path:
    p = tmp_path / name
    p.write_text(textwrap.dedent(text).lstrip())
    return p


def header(response, name: str) -> str:
    """*response*'s header, failing the test if the server left it out — which every assertion below would rather say plainly than pass ``None`` on."""
    value = response.getheader(name)
    assert value is not None, f"the response carries no {name}"
    return value


class TestParse:
    def test_header_and_string_prose(self, tmp_path):
        doc = parse(
            write(
                tmp_path,
                '''
                # title: T
                # code: hide

                """
                # H
                """

                # a comment belongs to the cell below
                x = 1
                f"""
                after {x}
                """
                y = 2
                ''',
            )
        )
        assert doc.meta == {"title": "T", "code": "hide"}
        assert doc.title == "T" and not doc.show_code
        assert doc.segments[:2] == (Prose("# H\n", 4), Cell("# a comment belongs to the cell below\nx = 1\n", 8))
        after, last = doc.segments[2:]
        assert (
            isinstance(after, Prose) and after.text == "after {x}\n" and after.line == 10 and after.fstring is not None
        )
        assert last == Cell("y = 2\n", 13)

    def test_title_from_h1_and_display_fences_are_prose(self, tmp_path):
        doc = parse(
            write(
                tmp_path,
                '''
                """
                # The title

                ```python
                not_a_cell = True
                ```
                """
                ''',
            )
        )
        assert doc.title == "The title"
        assert doc.cells == []
        assert isinstance(doc.segments[0], Prose) and "not_a_cell" in doc.segments[0].text

    def test_strings_in_code_are_not_prose(self, tmp_path):
        doc = parse(
            write(
                tmp_path,
                '''
                def f():
                    """a docstring"""
                    return "# %% not special"

                s = """
                nor this
                """
                r"""prose with \\(x\\)"""
                ''',
            )
        )
        assert [type(s).__name__ for s in doc.segments] == ["Cell", "Prose"]
        assert isinstance(doc.segments[0], Cell) and doc.segments[0].source.count('"""') == 4
        assert doc.segments[1] == Prose("prose with \\(x\\)\n", 8)

    def test_is_literate_script_needs_the_title_header(self, tmp_path):
        assert parse(write(tmp_path, '# title: T\n\n"""p"""\n', name="lit.py")).title == "T"
        assert is_literate_script(tmp_path / "lit.py")
        assert not is_literate_script(write(tmp_path, "x = 1\n", name="mod.py"))
        assert not is_literate_script(write(tmp_path, "# title: T\n", name="notes.md"))


class TestWeave:
    def test_interpolation_display_and_stdout(self, tmp_path):
        p = write(
            tmp_path,
            '''
            # code: show
            xs = [1, 2, 3]
            print("hello")
            "*shown*"
            rf"""
            Total {sum(xs)}.
            {"\\n".join(f"- item {x}" for x in xs)}
            """
            ''',
        )
        w = Runner(p).weave()
        assert w.errors == [] and not w.stopped
        assert "```python\nxs = [1, 2, 3]" in w.markdown
        assert '<pre class="stdout">hello\n</pre>' in w.markdown
        assert "*shown*" in w.markdown
        assert "Total 6." in w.markdown
        assert "- item 1\n- item 2\n- item 3\n" in w.markdown

    def test_fields_are_evaluated_and_the_text_dedented(self, tmp_path):
        p = write(
            tmp_path,
            '''
            # title: F

            x = 3.14159
            names = ["a", "b"]

            rf"""
                x is {x:.2f}, names are {", ".join(names)}, spec {x:{"." + str(1) + "f"}}, braces {{literal}}.

                - {names[0]!r}
            """

            y = x * 2

            f"""y is {y:.1f}."""
            ''',
        )
        doc = parse(p)
        assert [type(s).__name__ for s in doc.segments] == ["Cell", "Prose", "Cell", "Prose"]
        seg = doc.segments[1]
        assert isinstance(seg, Prose) and seg.fstring is not None
        assert (
            seg.text
            == "x is {x:.2f}, names are {', '.join(names)}, spec {x:{'.' + str(1) + 'f'}}, braces {literal}.\n\n- {names[0]!r}\n"
        )
        w = Runner(p).weave(doc)
        assert w.errors == [] and w.cells_run == 2
        assert "x is 3.14, names are a, b, spec 3.1, braces {literal}.\n\n- 'a'\n" in w.markdown
        assert "y is 6.3." in w.markdown

    def test_plain_strings_are_static(self, tmp_path):
        p = write(tmp_path, 'x = 1\n"""\nbraces {x} and {{ x }} stay as written.\n"""\n')
        w = Runner(p).weave()
        assert w.errors == [] and "braces {x} and {{ x }} stay as written.\n" in w.markdown

    def test_stdout_capture_is_per_thread(self, tmp_path, capsys):
        p = write(
            tmp_path,
            """
            import threading
            done = threading.Event()
            t = threading.Thread(target=lambda: (print("other thread"), done.set()))
            t.start()
            done.wait()
            t.join()
            print("mine")
            """,
        )
        w = Runner(p).weave()
        assert w.errors == [] and w.outputs[0].stdout == "mine\n"
        assert "other thread" in capsys.readouterr().out

    def test_code_hidden_by_default(self, tmp_path):
        p = write(tmp_path, 'secret = 1\nf"""after {secret}"""\n')
        md = Runner(p).weave().markdown
        assert "secret = 1" not in md and "after 1" in md

    def test_stop_renders_rest_with_pending_marks(self, tmp_path):
        p = write(
            tmp_path,
            '''
            known = 1
            res = None
            if res is None:
                stop("_pending_")
            summary = res["x"]
            rf"""
            ## Results

            Value {summary.m:.2f} and {fig(res)}, known {known} and {known + summary}.
            """
            raise RuntimeError("must not run")
            ''',
        )
        w = Runner(p).weave()
        assert w.stopped and w.errors == []
        assert "_pending_" in w.markdown
        assert "## Results" in w.markdown
        assert (
            'Value <mark class="pending">summary.m:.2f</mark> and <mark class="pending">fig(res)</mark>, '
            'known 1 and <mark class="pending">known + summary</mark>.' in w.markdown
        )
        assert len(w.outputs) == 1

    def test_pending_marks_survive_code_spans_and_fences(self, tmp_path):
        p = write(
            tmp_path,
            '''
            stop()
            rf"""
            The best is `{best.name}` and {best.mean}.

            ~~~python
            x = {best.mean}
            ~~~
            """
            ''',
        )
        out = to_html(Runner(p).weave().markdown)
        assert out.count('<mark class="pending">best.') == 3
        assert "&lt;mark" not in out and "litpending" not in out
        assert '<code><mark class="pending">best.name</mark></code>' in out

    def test_error_points_at_script_line_and_stops(self, tmp_path):
        p = write(tmp_path, '"""intro"""\n\nx = 1\n1 / 0\nf"""after {x}"""\n')
        w = Runner(p).weave()
        assert len(w.errors) == 1
        assert f'File "{p}", line 4' in (w.errors[0].error or "")
        assert "ZeroDivisionError" in w.markdown
        assert "after 1" in w.markdown  # x was bound before the error, prose still renders

    def test_a_displayable_value_mid_cell_is_an_error(self, tmp_path):
        """Only a cell's last expression is shown, so a figure or HTML string produced above the end would vanish; the runner says so instead. A side-effecting call that returns nothing is fine."""
        p = write(tmp_path, '"""intro"""\n\nprint("side effect")\n"".join(["<b>lost</b>"])\nx = 1\nf"""after {x}"""\n')
        w = Runner(p).weave()
        assert len(w.errors) == 1
        assert f'File "{p}", line 4' in (w.errors[0].error or "")
        assert "last statement of its cell" in w.markdown
        assert "<b>lost</b>" not in w.markdown

        p = write(tmp_path, '"""intro"""\n\nx = [1]\nx.append(2)\nx\n')
        w = Runner(p).weave()
        assert w.errors == [] and "[1, 2]" in w.markdown

    def test_a_failing_field_is_an_error_at_the_script_line(self, tmp_path):
        p = write(tmp_path, 'x = "s"\n\nf"""bad {x:.2f} {nope}"""\n')
        w = Runner(p).weave()
        assert w.errors == []  # prose errors are shown in place rather than stopping the script
        assert 'class="error"' in w.markdown and "doc.py:3: ValueError" in w.markdown

    def test_dataclass_and_inspect_work_in_cells(self, tmp_path):
        p = write(
            tmp_path,
            """
            import inspect
            from dataclasses import dataclass

            @dataclass
            class C:
                a: int

            def f():
                return 1

            inspect.getsource(f).strip()
            """,
        )
        w = Runner(p).weave()
        assert w.errors == []
        assert "def f():\n    return 1" in w.markdown

    def test_figure_is_saved_through_the_publisher(self, tmp_path):
        pytest.importorskip("matplotlib")
        p = write(tmp_path, "import matplotlib.pyplot as plt\nfig, ax = plt.subplots()\nfig\n")
        pub = Publisher(asset_dir=tmp_path / "_assets")
        md = Runner(p, publish=pub).weave().markdown
        assert '<img src="_assets/cell-0.png"' in md
        assert (tmp_path / "_assets" / "cell-0.png").exists()


class TestIncremental:
    def test_only_changed_suffix_reruns(self, tmp_path):
        p = write(tmp_path, 'log = []\na = 1\n"""\n"""\nlog.append("b")\nb = a + 1\nf"""b={b}"""\n')
        r = Runner(p)
        assert r.weave().cells_run == 2
        # prose-only edit: nothing re-runs
        p.write_text(p.read_text().replace("b={b}", "B={b}"))
        w = r.weave()
        assert w.cells_run == 0 and "B=2" in w.markdown
        # edit the second cell: only it re-runs, against the namespace after the first
        p.write_text(p.read_text().replace("b = a + 1", "b = a + 10"))
        w = r.weave()
        assert w.cells_run == 1 and "B=11" in w.markdown
        # edit the first cell: both re-run
        p.write_text(p.read_text().replace("a = 1", "a = 2"))
        w = r.weave()
        assert w.cells_run == 2 and "B=12" in w.markdown

    def test_functions_defined_earlier_see_the_restored_namespace(self, tmp_path):
        p = write(tmp_path, 'k = 1\ndef f():\n    return k\n"""\n"""\nout = f()\nf"""{out}"""\n')
        r = Runner(p)
        r.weave()
        p.write_text(p.read_text().replace("out = f()", "out = f() + 1"))
        assert r.weave().markdown.strip() == "2"


class TestPartial:
    def test_snapshot_before_each_cell_that_runs(self, tmp_path):
        p = write(
            tmp_path,
            '''
            """
            # Title
            """
            a = 1
            f"""a is {a}."""
            b = a + 1
            f"""b is {b}."""
            ''',
        )
        seen = []
        r = Runner(p)
        r.weave(partial=seen.append)
        assert [w.running.line for w in seen if w.running] == [4, 6]
        first, second = (w.markdown for w in seen)
        assert "# Title" in first and "Running the cell at line 4" in first
        assert 'a is <mark class="pending">a</mark>.' in first and 'b is <mark class="pending">b</mark>.' in first
        assert "a is 1." in second and "Running the cell at line 6" in second and "<mark" in second
        # cached cells do not run, so no snapshot is taken for them
        seen.clear()
        r.weave(partial=seen.append)
        assert seen == []


class TestMemo:
    @pytest.fixture(autouse=True)
    def cache(self, tmp_path):
        set_cache_dir(tmp_path / "cache")
        yield
        set_cache_dir(None)

    def test_hit_across_processes_and_miss_on_source_change(self, tmp_path):
        p = write(
            tmp_path,
            "from mini.lit import memo\ncalls = []\n@memo\ndef f(x):\n    calls.append(x)\n    return x * 2\nf(3), f(3), len(calls)\n",
        )
        assert "(6, 6, 1)" in Runner(p).weave().markdown
        # a second runner (a fresh process, as far as the memo is concerned) hits the disk cache
        set_cache_dir(tmp_path / "cache")  # clears the in-memory tier
        assert "(6, 6, 0)" in Runner(p).weave().markdown
        p.write_text(p.read_text().replace("x * 2", "x * 3"))
        assert "(9, 9, 1)" in Runner(p).weave().markdown

    def test_memo_key_stands_in_for_the_object(self, tmp_path):
        """A results object keyed by ``__memo_key__`` hits while the key holds and misses when it moves, whatever its other fields do."""
        src = (
            "from dataclasses import dataclass\nfrom mini.lit import memo\nimport numpy as np\ncalls = []\n"
            "@dataclass(frozen=True)\nclass Results:\n    sha: int\n    big: object\n    def __memo_key__(self):\n        return self.sha\n"
            "@memo\ndef f(res):\n    calls.append(1)\n    return res.sha\n"
            "f(Results(1, object())), f(Results(1, np.zeros(3))), f(Results(2, object())), len(calls)\n"
        )
        assert "(1, 1, 2, 2)" in Runner(write(tmp_path, src)).weave().markdown

    def test_miss_on_a_design_constant_read_off_a_module(self, tmp_path):
        """A report reads its gates as ``ex.GATE``; editing one must redraw what quotes it, though the task fingerprint alone would not see it."""
        design = write(tmp_path, "GATE = 0.02\n", name="design.py")
        p = write(
            tmp_path,
            "import design\nfrom mini.lit import memo\ncalls = []\n@memo\ndef f():\n    calls.append(1)\n    return design.GATE\nf(), len(calls)\n",
        )
        assert "(0.02, 1)" in Runner(p).weave().markdown
        set_cache_dir(tmp_path / "cache")
        assert "(0.02, 0)" in Runner(p).weave().markdown
        design.write_text("GATE = 0.03\n")
        # The rewrite kept design.py's size and landed in the same whole second, so Python's cached
        # bytecode still validates and the re-import would hand back the old GATE. Date it forward.
        os.utime(design, (later := design.stat().st_mtime + 2, later))
        set_cache_dir(tmp_path / "cache")
        sys.modules.pop("design")  # a fresh process would not hold the old module
        assert "(0.03, 1)" in Runner(p).weave().markdown

    def test_array_globals_are_evidence_and_opaque_ones_warn(self, tmp_path, caplog):
        """A figure that reads a module-level array must redraw when the array changes; one that reads something the cache cannot fingerprint is told to take it as an argument."""
        src = "import numpy as np\nfrom mini.lit import memo\nA = np.arange(3)\ncalls = []\n@memo\ndef f():\n    calls.append(1)\n    return int(A.sum())\nf(), len(calls)\n"
        p = write(tmp_path, src)
        assert "(3, 1)" in Runner(p).weave().markdown
        set_cache_dir(tmp_path / "cache")
        assert "(3, 0)" in Runner(p).weave().markdown
        p.write_text(src.replace("arange(3)", "arange(4)"))
        set_cache_dir(tmp_path / "cache")
        assert "(6, 1)" in Runner(p).weave().markdown

        q = write(
            tmp_path,
            "from mini.lit import memo\nclass Box: ...\nB = Box()\n@memo\ndef g():\n    return 1 if B else 0\ng()\n",
            name="opaque.py",
        )
        with caplog.at_level("WARNING", logger="mini.lit.caching"):
            Runner(q).weave()
        assert "g reads B (Box)" in caplog.text

    def test_array_inputs_key_by_content(self):
        np = pytest.importorskip("numpy")
        calls = []

        @memo
        def g(a):
            calls.append(1)
            return float(a.sum())

        assert g(np.arange(3)) == g(np.arange(3)) == 3.0
        assert len(calls) == 1
        g(np.arange(4))
        assert len(calls) == 2

    def test_hit_requires_its_assets(self, tmp_path):
        pub = Publisher(asset_dir=tmp_path / "_assets")
        calls = []

        @memo
        def h():
            calls.append(1)
            return pub.asset_url(b"png", name="fig.png")

        from mini.reports import use_publisher

        use_publisher(pub)
        try:
            assert h() == h() == "_assets/fig.png" and len(calls) == 1
            (tmp_path / "_assets" / "fig.png").unlink()
            h()
            assert len(calls) == 2 and (tmp_path / "_assets" / "fig.png").exists()
        finally:
            use_publisher(None)


class TestCellMark:
    def test_a_marker_splits_a_cell_and_both_halves_display(self, tmp_path):
        """Two values back to back, with no paragraph between: the marker is the boundary prose would have been."""
        p = write(tmp_path, '"""A."""\nx = "<b>one</b>"\nx\n# %%\ny = "<b>two</b>"\ny\n"""B."""\n')
        doc = parse(p)
        cells = [s for s in doc.segments if isinstance(s, Cell)]
        assert [c.line for c in cells] == [2, 5]
        assert "# %%" not in cells[0].source + cells[1].source
        r = render(p, out_dir=tmp_path / "out")
        assert r.woven.errors == [] and r.woven.markdown.count("<b>") == 2

    def test_a_marker_at_the_edges_or_alone_makes_no_empty_cell(self, tmp_path):
        p = write(tmp_path, '# %%\n"""A."""\n# %%\n\n# %%\nx = 1\n# %%\n')
        cells = [s for s in parse(p).segments if isinstance(s, Cell)]
        assert [(c.line, c.source) for c in cells] == [(6, "x = 1\n")]


class TestRender:
    def test_writes_html_and_markdown(self, tmp_path):
        p = write(tmp_path, '"""\n# Hi\n"""\nv = 2\nrf"""v is {v} and \\(x^2\\)."""\n')
        r = render(p, out_dir=tmp_path / "out")
        html = (tmp_path / "out" / "index.html").read_text()
        assert "<title>Hi</title>" in html and "v is 2" in html
        assert "katex" in html  # math present → KaTeX loaded
        assert "v is 2" in (tmp_path / "out" / "index.md").read_text()
        assert r.woven.errors == []

    def test_writes_each_output_with_its_figures_beside_it(self, tmp_path):
        """The CLI writes the files it is given, by extension; one in another directory gets its own copy of ``_assets/``, since every output links the figures relatively."""
        from mini.lit.render import write_outputs

        p = write(tmp_path, '"""\n# Hi\n"""\nv = 2\nrf"""v is {v}."""\n')
        r = render(p, out_dir=tmp_path / "a", write=False)
        (tmp_path / "a" / "_assets").mkdir()
        (tmp_path / "a" / "_assets" / "fig.png").write_bytes(b"png")
        written = write_outputs(r, [tmp_path / "a" / "r.md", tmp_path / "b" / "r.html"])
        assert [w.name for w in written] == ["r.md", "r.html"]
        assert "v is 2" in (tmp_path / "a" / "r.md").read_text()
        assert "<title>Hi</title>" in (tmp_path / "b" / "r.html").read_text()
        assert (tmp_path / "b" / "_assets" / "fig.png").read_bytes() == b"png"
        assert not (tmp_path / "a" / "index.html").exists(), "wrote the default page as well as the ones asked for"
        with pytest.raises(ValueError, match="unknown format"):
            write_outputs(r, [tmp_path / "a" / "r.txt"])

    def test_a_pdf_gets_no_copy_of_the_figures(self, tmp_path, monkeypatch):
        """The PDF is printed from the weave's own ``_assets/``, so a copy beside it would only pile up (every review print goes to one ``.mini/prints/``)."""
        import mini.report_print
        from mini.lit.render import write_outputs

        calls = []
        monkeypatch.setattr(
            mini.report_print, "print_bundle", lambda bundle, out, **kw: calls.append((bundle, out)) or out
        )
        p = write(tmp_path, '"""\n# Hi\n"""\nv = 2\nrf"""v is {v}."""\n')
        r = render(p, out_dir=tmp_path / "a", write=False)
        (tmp_path / "a" / "_assets").mkdir()
        (tmp_path / "a" / "_assets" / "fig.png").write_bytes(b"png")
        write_outputs(r, [tmp_path / "prints" / "r.pdf"])
        assert calls == [(tmp_path / "a" / "r.pdf", tmp_path / "prints" / "r.pdf")]
        assert not (tmp_path / "prints" / "_assets").exists()

    def test_markdown_links_an_svg_figure_where_the_page_inlines_it(self, tmp_path):
        p = write(
            tmp_path,
            'from mini.vis import svg_figure\n"""# T"""\nsvg_figure(\'<svg xmlns="http://www.w3.org/2000/svg"><path d="M0 0"/></svg>\', alt_text="a strip", name="strip")\n',
        )
        r = render(p, out_dir=tmp_path / "out")
        assert r.woven.errors == []
        assert "<path" in (tmp_path / "out" / "index.html").read_text()
        md = (tmp_path / "out" / "index.md").read_text()
        assert "<path" not in md and "[a strip](_assets/strip.html)" in md
        assert (tmp_path / "out" / "_assets" / "strip.html").exists()

    def test_page_carries_the_base_sheet_the_frame_and_the_report_styles(self, tmp_path):
        """One stylesheet stack however a page is made: base.css (shared with the site), lit.css (the frame),
        and the project's docs/report.css last, found beside the report tree — so `./go serve` shows a report.css
        edit live, and the export needs no step of its own."""
        (tmp_path / "pyproject.toml").touch()
        (tmp_path / "docs" / "report.css").parent.mkdir()
        (tmp_path / "docs" / "report.css").write_text("main.lit { color: rebeccapurple }")
        p = write(tmp_path / "docs", '"""# Hi"""\n')
        html = render(p, out_dir=tmp_path / "out").html
        assert "--font-mono:" in html and "main.lit {" in html  # base, then the frame
        assert html.index("--font-mono:") < html.index("main.lit {") < html.index("rebeccapurple")

    def test_live_output_is_a_separate_tree(self, tmp_path, monkeypatch):
        from mini.lit.render import output_dir

        (tmp_path / "pyproject.toml").touch()
        monkeypatch.chdir(tmp_path)
        doc = tmp_path / "docs" / "foo" / "report.py"
        assert output_dir(doc) == tmp_path / ".mini" / "lit" / "foo"
        assert output_dir(doc, live=True) == tmp_path / ".mini" / "lit-live" / "foo"

    def test_page_without_math_skips_katex(self):
        assert "katex" not in page(to_html("plain"), title="t")

    def test_markdown_dialect(self):
        html = to_html("/// admonition | T\n    type: note\nbody\n///\n\nx[^1]\n\n[^1]: note\n\n| a |\n|---|\n| 1 |\n")
        assert 'class="admonition note"' in html
        assert 'class="footnote' in html
        assert "<table>" in html


class TestSiblingImports:
    def test_each_script_imports_its_own_sibling(self, tmp_path):
        """Two reports each with an ``experiment.py`` beside them, run in one process: each sees its own."""
        ws = []
        for key in ("a", "b"):
            d = tmp_path / key
            d.mkdir()
            (d / "experiment.py").write_text(f"NAME = {key!r}\n")
            ws.append(write(d, "# title: T\n\nimport experiment as ex\n\nex.NAME\n"))
        assert "a" in Runner(ws[0]).weave().markdown
        assert "b" in Runner(ws[1]).weave().markdown
        assert "a" in Runner(ws[0]).weave().markdown  # and back again: the earlier directory moves to the front


class TestServe:
    """The live server's transport: what a browser (and whatever proxies for it) sees."""

    def _serve(self, tmp_path):
        import threading
        from functools import partial as bind

        from mini.lit.serve import _Handler, _Server, _Site

        out = tmp_path / "out"
        (out / "_assets").mkdir(parents=True)
        (out / "index.html").write_text("<p>hi</p>")
        (out / "_assets" / "fig.png").write_bytes(b"\x89PNG" + bytes(200_000))
        site = _Site(out)
        handler = type("Handler", (_Handler,), {"site": site})
        server = _Server(("127.0.0.1", 0), bind(handler, directory=str(out)))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return server, site

    def test_one_connection_serves_the_whole_page(self, tmp_path):
        """Keep-alive, and every body as long as its ``Content-Length`` says.

        A report asks for dozens of figures at once. Under HTTP/1.0 each is its own connection, and a connection lost in that burst reaches the browser as ``ERR_CONTENT_LENGTH_MISMATCH``.
        """
        import http.client

        server, _ = self._serve(tmp_path)
        try:
            conn = http.client.HTTPConnection(*server.server_address)
            for path in ("/", "/_assets/fig.png?v=deadbeef", "/index.html"):
                conn.request("GET", path)
                r = conn.getresponse()
                body = r.read()
                assert r.status == 200
                assert len(body) == int(header(r, "Content-Length"))
                assert r.version == 11 and not r.will_close  # the next request reuses this socket
        finally:
            conn.close()
            server.shutdown()

    def test_what_the_browser_may_keep(self, tmp_path):
        """A stamped figure is immutable, the page is revalidated, the poll and a miss are never kept."""
        import http.client

        server, _ = self._serve(tmp_path)
        try:
            conn = http.client.HTTPConnection(*server.server_address)
            got = {}
            for path in ("/", "/index.html", "/_assets/fig.png?v=deadbeef", "/_assets/fig.png", "/nope.png"):
                conn.request("HEAD", path)
                r = conn.getresponse()
                r.read()
                got[path] = r.getheader("Cache-Control")
            assert got["/"] == got["/index.html"] == "no-cache"
            assert got["/_assets/fig.png?v=deadbeef"] == "public, max-age=31536000, immutable"
            assert got["/_assets/fig.png"] == "no-cache"  # unstamped: the URL says nothing about the bytes
            assert got["/nope.png"] == "no-store"
        finally:
            conn.close()
            server.shutdown()

    def test_the_page_is_kept_until_a_build_changes_it(self, tmp_path):
        """A reload with no build behind it answers 304, so the browser keeps the page it parsed (DevTools with it)."""
        import http.client

        server, site = self._serve(tmp_path)
        try:
            conn = http.client.HTTPConnection(*server.server_address)
            conn.request("GET", "/index.html")
            r = conn.getresponse()
            r.read()
            etag = header(r, "ETag")
            conn.request("GET", "/index.html", headers={"If-None-Match": etag})
            r = conn.getresponse()
            assert r.status == 304 and r.read() == b""
            (tmp_path / "out" / "index.html").write_text("<p>a build landed</p>")
            conn.request("GET", "/index.html", headers={"If-None-Match": etag})
            r = conn.getresponse()
            assert r.status == 200 and r.read() == b"<p>a build landed</p>"
        finally:
            conn.close()
            server.shutdown()

    def test_a_replaced_file_mid_request_is_still_whole(self, tmp_path):
        """A build lands while the page is loading: the browser gets one version or the other, never a short body."""
        import http.client

        server, _ = self._serve(tmp_path)
        try:
            conn = http.client.HTTPConnection(*server.server_address)
            conn.request("GET", "/_assets/fig.png")
            r = conn.getresponse()
            (tmp_path / "out" / "_assets" / "fig.png").write_bytes(b"\x89PNG" + bytes(10))
            assert len(r.read()) == int(header(r, "Content-Length"))
        finally:
            conn.close()
            server.shutdown()

    def test_the_version_poll_answers_when_a_build_lands(self, tmp_path):
        import http.client
        import threading

        server, site = self._serve(tmp_path)
        try:
            threading.Timer(0.2, lambda: site.publish(lambda reload: "<p>next</p>" + reload)).start()
            conn = http.client.HTTPConnection(*server.server_address)
            conn.request("GET", "/__version?after=0")
            r = conn.getresponse()
            assert r.read() == b"1"
        finally:
            conn.close()
            server.shutdown()


class TestLazyNpz:
    @pytest.fixture(autouse=True)
    def cache(self, tmp_path):
        set_cache_dir(tmp_path / "cache")
        yield
        set_cache_dir(None)

    def test_reads_on_demand_and_keys_by_content(self, tmp_path):
        np = pytest.importorskip("numpy")
        path = tmp_path / "a.npz"
        np.savez_compressed(path, x=np.arange(3), y=np.ones((2, 2)))
        z = read_npz(path)
        assert z is not None and read_npz(None) is None
        assert sorted(z) == ["x", "y"] and len(z) == 2 and "x" in z and "q" not in z
        assert z["x"].tolist() == [0, 1, 2] and z["x"] is z["x"]  # decompressed once, then kept
        assert z.__memo_key__() == LazyNpz(path.read_bytes()).__memo_key__()
        np.savez_compressed(path, x=np.arange(4))
        assert LazyNpz(path.read_bytes()).__memo_key__() != z.__memo_key__()

    def test_memo_takes_it_as_an_argument(self, tmp_path):
        np = pytest.importorskip("numpy")
        path = tmp_path / "a.npz"
        np.savez_compressed(path, x=np.arange(3))
        calls = []

        @memo
        def total(z: LazyNpz) -> int:
            calls.append(1)
            return int(z["x"].sum())

        z = LazyNpz(path.read_bytes())
        assert total(z) == 3 and total(LazyNpz(path.read_bytes())) == 3 and len(calls) == 1
