"""The model: nGPT, the normalized transformer on the unit hypersphere.

The `ngpt` module tells the architecture's story end to end; the primitives it
builds on (and the `LanguageModel` base) live in `_shared`.
"""

from jaxtyping import PRNGKeyArray

from sca.config import ModelConfig
from sca.model._shared import LanguageModel
from sca.model.ngpt import NGPT

__all__ = ["NGPT", "LanguageModel", "build_model"]


def build_model(config: ModelConfig, *, key: PRNGKeyArray) -> NGPT:
    """Construct the model."""
    return NGPT(config, key=key)
