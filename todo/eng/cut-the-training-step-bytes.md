---
status: open
tags: [performance, training]
opened: 2026-10-31
---
# Cut the training step's bytes

The pack HLO scout (`docs/m2/pack-hlo`, write-up in the project files as `pack-timing/pack-hlo-findings.md`) found that one d64 seed's step on an L40S runs mostly out of the 96 MB L2 cache: its working set is about 92 MB, its largest kernels move data at about 3.2 TB/s against the 864 GB/s main memory can supply, and a pack of seeds is slower per seed because it spills to main memory. So a single run is close to the cache's rate, and what is left to gain on one seed comes from moving fewer bytes or launching fewer kernels. Two candidates, each worth about a tenth of the step:

- A fused attention kernel. The (64, 8, 96, 96) attention maps are 19 MB per layer, and the kernels that read or write them are a sixth of one seed's kernel time. cuDNN's fused attention takes 16-bit floats only, so in float32 this would be a Pallas or Triton kernel. It is a numerics change.
- Kernel launches. A step runs about 490 kernels, and the card is idle about 0.26 ms of the 1.8 ms step between them (flags off). XLA can capture runs of kernels as CUDA graphs (command buffers); whether the scanned step is captured, and what a wider capture would save, is untested. The `--xla_gpu_enable_command_buffer` flag and the scout's profiler trace are the place to start.

Neither changes the conclusion that seed packs don't pay on the L40S: the other large kernels (MLP hidden state, logits) also slow down about 1.5-fold in a pack of 8.
