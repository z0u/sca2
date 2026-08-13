---
status: open
tags: [publishing]
opened: 2026-08-06
---
# The publish check only sees changed notebooks

`scripts/unpublished_reports.py` compares a git diff against `publish.lock`, so a report whose inputs moved while its `report.py` didn't — a re-run experiment writing new refs, a restyle in `src/sca/vis*.py`, an edit to `docs/report.css` — passes the check while serving stale figures. A sibling `experiment.py` change is the cheap 80% heuristic (same directory, likely new results); the sharper version compares the store refs the last export resolved (`_assets/provenance.json` already records them) against what's current. Worth doing once we notice it biting.
