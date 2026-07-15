#!/usr/bin/env bash

set -euo pipefail

SELF="${BASH_SOURCE[0]}"
PROJECT_ROOT="$( cd -- "$( dirname -- "$SELF" )" &> /dev/null && pwd )"
SCRIPT_DIR="$PROJECT_ROOT/scripts"

is_marimo_notebook() {
    [[ "${1:-}" == *.py && -f "${1:-}" ]] && grep -q 'marimo\.App(' "$1"
}

show_usage() {
    # Important: heredoc indented with tab characters.
    cat <<-EOF
		Usage: $0 {install|auth|check|open|preview|publish|site|help}
		New checkout? Start with: $0 install

		  install:             install dependencies (uv sync) and git hooks
		  auth    [--check]:   set up credentials; --check just probes
		  check   [--lint] [--format] [--typecheck] [--test] [--fix]:
		                       run checks in parallel (default: all without --fix)
		                       individual commands: format | lint | types | tests | dead
		  open    <file>:      open a Marimo notebook in Marimo, or anything else in \$EDITOR
		  preview [...nbs] [--no-serve] [--force] [--port N]:
		                       export stale reports, assemble the site with local assets
		                       (never touches the network), and serve it
		  publish <nbs|--all>: export reports and sync their bundles to the publish tier
		  site:                assemble the public site from *published* bundles into _site/
		                       (for CI; read-only, never runs a notebook)

		Experiments are run with \`bin/mini\`, not \`$0\` — see \`bin/mini --help\`.
		EOF
}

case "${1:-help}" in
    i|install)
        shift
        "$SCRIPT_DIR/install.sh" "$@"
        ;;
    auth)
        shift
        "$SCRIPT_DIR/auth.sh" "$@"
        ;;
    format|formatting)
        shift
        "$SCRIPT_DIR/format.sh" "$@"
        ;;
    lint|linting|linters)
        shift
        "$SCRIPT_DIR/lint.sh" "$@"
        ;;
    dead|deadcode)
        shift
        "$SCRIPT_DIR/deadcode.sh" "$@"
        ;;
    type|types|typecheck)
        shift
        "$SCRIPT_DIR/typecheck.sh" "$@"
        ;;
    test|tests)
        shift
        "$SCRIPT_DIR/test.sh" "$@"
        ;;
    c|check)
        if [[ $# -gt 1 ]]; then
            shift
            "$SCRIPT_DIR/check.sh" "$@"
        else
            "$SCRIPT_DIR/check.sh" --lint --format --typecheck --test
        fi
        ;;
    o|edit|open)
        shift
        if [[ $# -eq 0 ]]; then
            echo "open what? pass a Marimo notebook (opens in marimo edit)," 1>&2
            echo "or any other file (opens in \$VISUAL/\$EDITOR)." 1>&2
            exit 2
        elif is_marimo_notebook "${1:-}"; then
            uv run marimo edit "$@"
        else
            editor="${VISUAL:-${EDITOR:-code}}"
            if ! command -v "$editor" > /dev/null; then
                echo "no editor: '$editor' not found — set \$VISUAL or \$EDITOR" 1>&2
                exit 127
            fi
            "$editor" "$@"
        fi
        ;;
    p|preview)
        shift
        serve=1 port=8000
        export_args=(--stale-only)
        while [[ $# -gt 0 ]]; do
            case "$1" in
                --no-serve) serve=0 ;;
                --force) export_args=() ;;
                --port) port="${2:?--port needs a value}"; shift ;;
                -*) echo "preview: unknown flag '$1' (flags: --no-serve --force --port N)" 1>&2; exit 2 ;;
                *) export_args+=("$1") ;;
            esac
            shift
        done
        ( set -x; uv run "$SCRIPT_DIR/export_reports.py" "${export_args[@]}" )
        ( set -x; uv run "$SCRIPT_DIR/build_site.py" --localize )
        if [[ $serve -eq 1 ]]; then
            ( set -x; uv run "$SCRIPT_DIR/preview_server.py" "$PROJECT_ROOT/_site" "$port" )
        else
            echo
            echo "Site assembled at _site/ (bundles in .mini/exports/)."
            echo "Serve it later with: $0 preview  — or render a bundle headlessly (report-render skill)."
        fi
        ;;
    publish)
        shift
        # Export each named report and mirror its bundle to the publish tier (needs ./go auth).
        # Explicit by design: export_reports.py refuses a bare --publish without names or --all.
        ( set -x; uv run "$SCRIPT_DIR/export_reports.py" --publish "$@" )
        ;;
    site)
        shift
        uv run "$SCRIPT_DIR/build_site.py" --externalize "$@"
        ;;
    e|export|r|run|s|serve|build|scrub|clean)
        case "$1" in
            e|export)     echo "'export' is gone — '$0 preview --no-serve' exports stale reports to .mini/exports/" ;;
            r|run)        echo "'run' is gone — 'bin/mini run <experiment.py>' runs experiments; '$0 preview' renders reports; 'uv run ...' for anything else" ;;
            s|serve)      echo "'serve' is gone — '$0 preview' exports what's stale, then builds and serves" ;;
            build)        echo "'build' split in two — '$0 preview' assembles locally; '$0 site' assembles the public site from published bundles (CI)" ;;
            scrub|clean)  echo "'scrub' is internal now (export applies it) — scripts/clean_docs.py if you really need it" ;;
        esac 1>&2
        exit 2
        ;;
    h|help|-h|--help)
        show_usage
        exit 0
        ;;
    *)
        show_usage 1>&2
        exit 1
        ;;
esac
