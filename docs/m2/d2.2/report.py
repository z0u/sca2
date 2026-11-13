# title: D2.2: figures for the post

from pathlib import Path

from mini.reports import current_publisher
from mini.vis import figure_html

r"""
# D2.2: figures for the post

Figures for the post on D2.2, drawn here so the post can upload them from the bundle of this page. This page is not an experiment. The results behind each figure are in the reports it names, which hold the claims and their evidence; the [design doc](./design.md) has the recipe the figures describe.

## Pooling the pull

The anchor term pulls the states of each labeled context toward the anchor axis. It pools that pull twice before averaging over the batch: a mellowmax over the positions of the context, and then a mellowmax over slices. The mellowmax is a soft minimum, so each pool lets the term choose which states to pull, within a context and across depth. The [ex-2.1.9 report](../ex-2.1.9/report.py) introduced the pool over positions and drew the first version of this diagram; the pool over slices is new in D2.2.
"""

# A static SVG beside this script, revised from the ex-2.1.9 one; the publisher copies it into the bundle's assets.
pooling_svg = Path(__file__).parent / "public" / "pooling.svg"
pooling_url = pub.asset_url(pooling_svg, name="pooling.svg") if (pub := current_publisher()) else str(pooling_svg)
figure_html(
    f'<img src="{pooling_url}" alt="A training batch drawn as a cube with axes for position, slice, and batch. Two labeled sequences are shaded. Of five slices, the middle three each have a magenta line pooling over the positions of a labeled sequence, a magenta line along the slice edge pools those results, and an orange line down the batch edge takes the mean.">',
    caption="The anchor term pools the pull over positions, then over slices, then averages over the batch.",
)
