---
status: open
tags: [reports, vis]
---
# Animated visualizations of how the model changes over training

I want to see how the model changes over the course of training. We have the trajectories, and they hint at the schedules playing a part in the final shape of the cube in latent space (rather than term weight being the sole determining factor). So, let's create animated visualizations of:

- Scatter plots of alignment, e.g. those titled "Alignment at op1, per color."
- The smooth-step per token metrics plots
- Linear probe readouts of the color cube, against the true cube

If this were interactive, I'd like to be able to scrub through epochs, conditions, and metrics. If the data required would be large, we'll need to consider some chunked format. This could be embedded in a Marimo notebook, but if it's easier to have it as a standalone SPA, that's fine too.
