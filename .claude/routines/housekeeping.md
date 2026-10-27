# Housekeeping

Branch: `claude/routine-housekeeping`. Modal: none. The thread always replies after this one, with the PR link (if any) and the current priority items.

## The helper's brief

Hi Opus-san! Let's do a little ✨project management✨. A tidy backlog lets the user, and every thread that starts from it, see at a glance what's next, and it saves redoing work that's already finished. Please look through the backlogs — sometimes we finish off work and forget to mark it done, etc.

- Check open PRs to avoid rework
- Use `./go todo` to list the unclosed items
- Look at recent closed PRs to see what has changed recently (but also consider older items that might be stale)
- Update `todo/**/*.md` as appropriate, including priorities
- Use dated footnotes to help future-you to avoid rework
- Prefix the PR title with `[PM]`

At first, there will probably be quite a bit to take care of, but don't do too much at once: just fold a couple of things. Later, there might not be much. Again, don't search too hard: if you don't see anything that needs changing, that's OK!

Your changes will come through as suggestions (because PR). But also report the current list of priority items (`./go todo --priority`). If anything about the task or the environment seems off, add a Concerns section to your report; it reaches the user word for word.
