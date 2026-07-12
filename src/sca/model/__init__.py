"""The model: nGPT, the normalized transformer on the unit hypersphere.

The `ngpt` module tells the architecture's story end to end; the primitives it
builds on (and the `LanguageModel` base with the sampling machinery) live in
`_shared`.
"""

from jaxtyping import PRNGKeyArray

from sca.config import ModelConfig
from sca.model._shared import Generation, LanguageModel, SingleGeneration
from sca.model.ngpt import NGPT

__all__ = ["NGPT", "Generation", "LanguageModel", "SingleGeneration", "build_model"]


def build_model(config: ModelConfig, *, key: PRNGKeyArray) -> NGPT:
    """Construct the model."""
    return NGPT(config, key=key)
