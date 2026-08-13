---
status: open
tags: [cli, devx, reports]
---
# An easier way to open reports for editing

What works:

1. Launch Marimo in edit mode: `marimo edit --watch --headless docs/.../report.py`
2. Open the editor by clicking on: `URL: http://localhost:2718?access_token=...`
3. Click the "Toggle app view" button in the Marimo UI
4. Edit the source file directly in the IDE instead of Marimo

This then allows humans an AIs to edit the source, and the human can see the changes in the Marimo UI. Preprequisite config (already applied):

```toml
[tool.marimo.runtime]
watcher_on_save = "autorun"
```

The friction is mostly in steps 1 and 3. 1, because it's a long command to have to type. I never use `./go open` or `./go edit` or whatever: they're not the right verbs. They can be retired an new ones added, if that's what makes sense. 3, because you have to find the button with your mouse. Can we add a query param to the URL to automate it? Having to click on the URL in 2 is a bit annoying; it's only necessary because of `--headless` — BUT without `--headless` the URL is opened in the external browser. I want it to open inside VS Code, which it does if I Cmd-click on it.
