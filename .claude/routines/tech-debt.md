# Tech debt

Branch: `claude/routine-tech-debt`. Modal: may be used to test functionality, but don't spend much on compute.

## The helper's brief

Hi Opus-san! Let's pay down some ✨tech debt✨. Small engineering improvements make every later experiment quicker and less error-prone, for the user, for you, and for the helpers that come after you. This is a bit of a choose your own adventure. You could:

- Look for low-hanging fruit in the engineering backlog — `./go todo eng` lists it, one file per item under `todo/eng/`. `--tag T` narrows it, and `--bundle B` groups items that one session should take together
- Contribute to an existing PR
- Refactor something
- Check GH Actions workflows, see if they are behaving well or could be improved

Pick the one that most interests you, or that you think will improve your future developer experience. If you're having trouble choosing, make a shortlist and use a script to pick at random. Choose something one run can finish, and use `./go check` (plus any test you add) to confirm it works.

When you finish an item, set `status: done` in its front matter and add a `closed:` date, in the same PR as the work. If you open a new PR, prefix the title with `[TD]`.

Only work on engineering stuff. There is science debt too, but that has its own routine and I want to be directly involved in the scheduling of it.

If nothing looks doable, that's OK! And if there's something that you would _like_ to work on but it's blocked, just let me know. If a tool or test looks broken, say so instead of working around it, and if anything about the task or the environment seems off, add a Concerns section to your report; it reaches the user word for word. Please report the PR link (if any) and anything you'd like to work on but is blocked.
