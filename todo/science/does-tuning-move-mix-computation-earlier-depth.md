---
status: open
tags: [D2.1, representations, ex-2.1.5]
---
# Does tuning move the mix computation earlier in depth?

Conjecture from ex-2.1.5 drafting: pretraining has no pressure to compute the result before the answer position, so the mix stays pressed against the last layer (ex-2.1.4 saw exactly this at L4); an instruction-tuned variant might instead show operands decodable mid-stack and results near the head. Relevant to anchoring because it changes how many layers a result-concept anchor can act on.
