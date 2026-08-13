---
status: open
tags: [experiments, storage]
---
# ex-2.9.3 publishes exemplar refs that no report reads

ex-2.9.3's `publish_results` publishes `exemplar-hot`/`exemplar-cool` refs (the worst catastrophic run and its cooled-LR rescue) that no report reads. Either add the intended before/after rescue figure to the ex-2.9.3 report (mirroring ex-2.9.2's exemplar plot) or drop the two `set_ref` calls and the `worst`/`rescue` computation.
