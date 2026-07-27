import os

# XLA:CPU's concurrency-optimized scheduler trades memory for parallelism,
# which lets the O(T²) attention buffers of every layer be live at once —
# easily exceeding the RAM of a modest dev machine. Prefer the memory-frugal
# scheduler. Takes effect when the XLA backend initializes (i.e. on first use
# of jax after this package is imported); no effect on GPU/TPU.
#
# `setdefault`, so it yields to an XLA_FLAGS already in the environment. Task
# workers get theirs from `[tool.mini] env` in pyproject.toml, which carries this
# same flag plus the GPU determinism pair — set at the container/subprocess level
# because by the time a *second* task in a reused container imports this module,
# the backend is already up and its flags are fixed (eng/determinism.md). What's
# left for this line is the interactive path: notebooks, scripts, tests.
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_enable_concurrency_optimized_scheduler=false")
