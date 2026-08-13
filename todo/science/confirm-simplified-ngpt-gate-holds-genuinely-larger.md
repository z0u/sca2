---
status: open
tags: [model-arch]
---
# Confirm the simplified nGPT gate holds at a larger size

Confirm the simplified nGPT gate holds at a genuinely larger size (wider/deeper than 128×12, bigger GPU + batch) before leaning on it for M3. ngpt-scaling shows the fixed scalar α = 1/n_layer trains flat across the width × depth grid we can afford.
