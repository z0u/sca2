---
status: open
tags: [D2.2, anchoring, optimizer]
opened: 2026-10-24
---
# An anchored arm under Muon

The [optimizer scout](/docs/m2/optimizer-scout/report.py) found that Muon learns the task as well as Adam at three times the rate its finder suggests, with no privileged basis. Two things would come first: a finer rate ladder (say 1.5×, 2× and 5× the finder rate), since the top rung did best at both seeds but the finder rate led for most of training; and term weights of its own, since under Muon a typical direction costs two to three times what a quiet Adam axis costs, so the anchor would have no quiet axis to take.

Questions it would answer: whether the anchor holds and removes as cleanly without a privileged basis, and whether an anchored color such as *red* grades differently (the scout found the color embeddings laid out much as under Adam, E4). Low priority while the project trains with Adam; worth it if later models are trained without Adam.
