---
name: report-render
description: View a report's figures. Read the matplotlib PNGs directly, or render inline/JS figures and the full page in a headless browser (offline, by bundling Marimo assets).
---

# Rendering a report to check it

## Fast path: read the figure PNGs directly (no browser)

Most report figures are **matplotlib**, and the `report_bundle` publisher
(`mini.reports` + the `themed`/`light_dark` vis helpers) writes each one to disk as
a real file — `_assets/<name>-light.png` / `-dark.png` — during the bundle build,
regardless of the surrounding HTML. So the ergonomic way to *see* those figures is
to build the bundle and `Read` the PNGs. No browser, no runtime, no network:

```bash
./go preview --no-serve docs/m2/ex-2.1.1/report.py   # -> .mini/exports/m2/ex-2.1.1/
ls .mini/exports/m2/ex-2.1.1/_assets/*.png           # then Read the ones you want
```

This covers the bulk of every current report. Reach for the browser below only for
figures that *aren't* standalone PNGs — inline SVG (e.g. `subline` sparklines),
future JS-rendered charts (altair/plotly) — or when you need the **whole page**
(prose + figures together, layout, the show-code toggle). Those live only inside
marimo's client-hydrated data island (JSON, unicode-escaped `<svg…`), so
there's no file to read and the page is blank until the runtime renders it.

A standalone `.svg` file (no marimo runtime involved) rasterizes to a readable PNG
without a browser via cairosvg — `libcairo`/`librsvg` are present in this env:

```bash
uv run --with cairosvg python -c "import cairosvg; cairosvg.svg2png(url='x.svg', write_to='x.png', scale=2)"
```

## Browser path: for inline/JS figures, full page, or DOM assertions

A `marimo export html` bundle loads its frontend runtime (~200 JS/CSS/font URLs)
from `cdn.jsdelivr.net/npm/@marimo-team/frontend@<version>/dist`. In a
network-restricted sandbox the browser can't reach that CDN, so the page stays
blank — you can't screenshot it, and DOM assertions see nothing.

The fix: the *same* pinned `dist/` ships inside the marimo pip package under
`_static/`. Repoint the bundle's CDN refs at those local assets, serve the result,
and drive the pre-installed Chromium. No network, real render.

`render.py` (beside this file) does the whole dance — build a serve root from
marimo's `_static/` plus the bundle's CDN-rewritten HTML, serve it, screenshot:

```bash
# Get a bundle first if you don't have one: ./go preview --no-serve docs/m2/ex-2.1.1/report.py
#   -> .mini/exports/m2/ex-2.1.1/  (index.html + _assets/)
uv run --with playwright python .claude/skills/report-render/render.py \
    .mini/exports/m2/ex-2.1.1 -o /tmp/report.png
```

Then `Read` the PNG. `--suffix '?show-code=true'` appends to the URL;
`--wait-text 'some heading'` blocks until that text renders instead of a fixed
timeout.

To inspect **one element** instead of the whole page, pass a CSS selector —
`render.py` shoots each match (numbering `out.png` → `out-0.png`, `out-1.png`, …
when several match) after scrolling it into view:

```bash
uv run --with playwright python .claude/skills/report-render/render.py \
    .mini/exports/m2/ex-2.1.1 --selector '.output svg' -o /tmp/fig.png
```

`.output` wraps each Marimo cell's rendered output, so `.output svg` targets the
report's inline figures (`.output img` for rasterized ones, `.output table` for a
dataframe). Tighter than a full-page shot when you only care about one figure.

## Asserting on behavior, not just looking

For toggles / visibility / layout logic, drive the DOM instead of screenshotting.
`render.py`'s `_build_serve_root` + `_serve` are the reusable core; swap the
screenshot for Playwright queries. This is how the show-code default was pinned
down (PR #22) — e.g. across `?show-code` values:

```python
page.goto(f"http://127.0.0.1:{port}/index.html?show-code=false")
page.wait_for_timeout(3500)
code_shown = page.evaluate("document.body.innerText.includes('import marimo')")
page.locator("[aria-haspopup=menu]").first.click()          # open the ⋮ menu
toggle = page.locator("[data-testid=notebook-action-show-code]").count()
```

## Why it works / gotchas

- **Run through the project env** (`uv run --with playwright`), *not* `uvx`: the
  local `_static/` assets are hash-named per marimo version, so they only match a
  bundle exported by the *same* marimo. `uvx --with marimo` would resolve some
  other version and every asset would 404.
- Two asset dirs, no collision: the runtime lives under `assets/` (from
  `_static/`), the report's figures under `_assets/` (leading underscore, from the
  bundle). `render.py` copies both into the serve root — copies, not symlinks, so
  a write into the serve root can never reach through a link and mutate the marimo
  package or the bundle (a symlinked `index.html` once let exactly that happen).
- **Chromium:** in the Claude-on-web sandbox it's pre-baked at
  `/opt/pw-browsers/chromium` — `render.py` uses that if present. In VS Code / a
  fresh dev container it's *not* there (and `/opt/pw-browsers` isn't writable), so
  `render.py` falls back to Playwright's default resolution. Install it once:
  ```bash
  uv run --with playwright playwright install chromium        # -> ~/.cache/ms-playwright
  uv run --with playwright playwright install-deps chromium   # OS libs (libxkbcommon0, …)
  ```
  A candidate for baking into the dev container if this becomes routine; on-demand
  is fine otherwise (one download, then cached).
- **Locale:** headless Chromium in a locale-less container reports no
  `navigator.language`, and marimo's frontend hard-errors on boot ("Incorrect
  locale information provided") — a blank-ish page with that message, not your
  report. `render.py` pins `locale="en-US"` on the page to avoid it.
- A missing favicon/font 404 is cosmetic — the app still renders.

This same repoint-CDN-to-`_static` trick is what a full offline/archival bundle
would do at publish time; here it's just scoped to a throwaway render.
