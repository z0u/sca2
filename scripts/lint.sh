#!/usr/bin/env bash

set -euo pipefail

( set -x; uv run ruff check "$@" )
( set -x; uv run "$(dirname "$0")/todo.py" --check )
( set -x; uv run "$(dirname "$0")/trailing_cell_docstrings.py" )
( set -x; uv run "$(dirname "$0")/prose_fields.py" )

echo "✅ Lint check passed"
