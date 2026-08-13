---
status: done
tags: [D2.1, ex-2.1.6]
---
# Is 100 epochs more than this testbed needs?

**Yes — 50 is enough.** Ex-2.1.11's `short` arm runs the whole recipe at half length with every keyframe scaled and lands within band on all five decision statistics (m_line +0.0014 against a band of 0.0144), at a task cost of 0.0013 against its own control. The survey stage runs at 50 epochs. Original note follows. Ex-2.1.6's margin is flat from roughly epoch 50 in every anchored condition, and validation loss settles earlier still, so half the schedule may be buying nothing. Purely a budget question — it does not interact with the alignment decay above, and it should be settled by a length arm rather than by reading a stopping point off runs we have already scored. Worth doing before the sweeps get wider in D2.2.
