#!/usr/bin/env python
"""Post-export tidy-ups for Marimo HTML and session JSON: collapse terminal control
sequences, redact URLs that shouldn't ship, and hide report code by default.
"""

import json
import re
import sys
from pathlib import Path

# Exported report bundles (gitignored). clean_html/clean_session_json are also imported
# by export_reports.py and applied at export time, before a bundle is synced.
EXPORTS_DIR = Path(__file__).parent.parent / ".mini" / "exports"

_CSI = re.compile(r"\x1b\[([0-9;?]*)([A-Za-z])")

REDACT: list[tuple[re.Pattern, str]] = [
    (re.compile(r"https://modal\.com/apps/\S+"), "[modal.com/apps/…]"),
]


def _needs_redaction(text: str) -> bool:
    return any(p.search(text) for p, _ in REDACT)


def _redact(text: str) -> str:
    """Redact patterns that should not appear in published output."""
    for pattern, replacement in REDACT:
        text = pattern.sub(replacement, text)
    return text


def _apply_terminal(text: str) -> str:  # noqa: C901
    """Collapse CR/erase/cursor-up sequences; keep SGR color codes."""
    lines: list[list[str]] = [[]]
    row = 0
    i = 0
    while i < len(text):
        c = text[i]
        if c == "\n":
            row += 1
            while len(lines) <= row:
                lines.append([])
            i += 1
        elif c == "\r":
            lines[row] = []
            i += 1
        elif c == "\x1b":
            m = _CSI.match(text, i)
            if m:
                params, cmd = m.group(1), m.group(2)
                if cmd == "m":
                    lines[row].append(m.group(0))  # SGR: keep color/style
                elif cmd == "K":
                    lines[row] = []  # erase line (any variant)
                elif cmd == "A":
                    n = int(params) if params and params.isdigit() else 1
                    row = max(0, row - n)
                elif cmd == "J" and params == "2":
                    lines, row = [[]], 0
                # all other CSI (cursor pos, mode, hide/show cursor) — drop
                i = m.end()
            else:
                i += 1  # bare or unrecognised ESC — skip
        else:
            lines[row].append(c)
            i += 1

    result = "\n".join("".join(line) for line in lines).strip()
    return re.sub(r"\n{3,}", "\n\n", result)


# --- HTML cleaning (regex-based, avoids parsing the JS wrapper) ---

_TEXT_FIELD = re.compile(r'("text":\s*")((?:[^"\\]|\\.)*)"')


def _has_control_seqs(raw: str) -> bool:
    return "\\r" in raw or "\\u001b" in raw


def clean_html(path: Path) -> bool:
    content = path.read_text("utf-8")

    def replace(m: re.Match) -> str:
        prefix, inner = m.group(1), m.group(2)
        if not _has_control_seqs(inner) and not _needs_redaction(inner):
            return m.group(0)
        try:
            text = json.loads(f'"{inner}"')
        except json.JSONDecodeError:
            return m.group(0)
        cleaned = _apply_terminal(text) if _has_control_seqs(inner) else text
        cleaned = _redact(cleaned)
        if cleaned == text:
            return m.group(0)
        return prefix + json.dumps(cleaned)[1:-1] + '"'

    new_content = _TEXT_FIELD.sub(replace, content)
    if new_content == content:
        return False
    path.write_text(new_content, "utf-8")
    return True


# --- Default to hidden code ---
#
# marimo's static export hardcodes the app view to *show* code — a global "Show code"
# toggle in the notebook menu that starts on. Our reports are literate (narrated prose
# and figures), so they read better with the code out of the way. Flip that initial
# state off; the toggle and the code both stay in the export, so a reader can still
# reveal it. Regex rather than a JSON parse because the config is embedded mid-page in
# a JS wrapper; a marimo format change simply no-ops here instead of corrupting output.

_SHOW_APP_CODE = re.compile(r'("showAppCode":\s*)true')


def hide_code_by_default(path: Path) -> bool:
    content = path.read_text("utf-8")
    new_content, n = _SHOW_APP_CODE.subn(r"\1false", content)
    if not n:
        return False
    path.write_text(new_content, "utf-8")
    return True


# --- Session JSON cleaning (proper JSON parse/dump) ---


def _clean_console(entries: list) -> bool:
    """Clean console entries in-place. Returns True if any changed."""
    changed = False
    for entry in entries:
        text = entry.get("text", "")
        if not text:
            continue
        needs_clean = "\r" in text or "\x1b" in text
        if not needs_clean and not _needs_redaction(text):
            continue
        cleaned = _apply_terminal(text) if needs_clean else text
        cleaned = _redact(cleaned)
        if cleaned != text:
            entry["text"] = cleaned
            changed = True
    return changed


def clean_session_json(path: Path) -> bool:
    data = json.loads(path.read_text("utf-8"))
    changed = False
    for cell in data.get("cells", []):
        if _clean_console(cell.get("console", [])):
            changed = True
    if not changed:
        return False
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", "utf-8")
    return True


# --- Main ---


def _clean(path: Path) -> bool:
    if path.suffix == ".html":
        # `|` not `or`: run both, don't short-circuit on the first that changes nothing.
        return clean_html(path) | hide_code_by_default(path)
    if path.name.endswith(".py.json"):
        return clean_session_json(path)
    return False


def main() -> None:
    if sys.argv[1:]:
        paths = [Path(a) for a in sys.argv[1:]]
    else:
        paths = list(EXPORTS_DIR.rglob("*.html")) + list(EXPORTS_DIR.rglob("*.py.json"))

    changed = 0
    for p in paths:
        if _clean(p):
            print(f"cleaned: {p}")
            changed += 1
        else:
            print(f"unchanged: {p}")
    print(f"{changed}/{len(paths)} files updated")


if __name__ == "__main__":
    main()
