from ftfy import ExplanationStep
from pydantic import BaseModel, NonNegativeFloat, NonNegativeInt, PositiveFloat, PositiveInt

from utils.param_types import IntX8, IntX32, IntX64, ZeroToOne


class ModelConfig(BaseModel, validate_assignment=True):
    vocab_size: IntX64
    """Vocabulary size"""

    block_size: IntX32
    """Maximum sequence length. A multiple of 32, so ex-2.2.15 can halve the usual 64."""

    n_embd: IntX8
    """Embedding dimension"""

    n_head: PositiveInt
    """Number of attention heads per layer.

    Positivity is the only constraint. Attention batches over this axis, so it
    stays clear of the matmul shapes that accelerator tiling cares about;
    `n_head_dim` is where the multiple-of-8 alignment earns its keep."""

    n_head_dim: IntX8
    """QKV dimension per-head, usually n_embd // n_head"""

    n_ff: IntX32
    """MLP dimensions, usually 4 * n_embd"""

    n_layer: PositiveInt
    """Number of transformer blocks; also sets the residual step size (1/n_layer)"""

    residual_alpha_exp: PositiveFloat = 1.0
    """Residual step exponent: alpha = n_layer ** -exp. 1.0 = 1/n_layer, 0.5 = 1/sqrt(n_layer)."""

    learnable_alpha: bool = False
    """Make the residual step a learnable scalar gain (init n_layer ** -exp) per sublayer,
    rather than a fixed constant."""

    tie_embeddings: bool = True
    """Share one table between the token embedding and the LM head (nGPT's default). With
    `False` the model carries a second `[V, C]` table for the readout, initialized as a copy
    of the embedding and trained separately from it."""

    line_mask_token: NonNegativeInt | None = None
    """Stop attention at this token (the newline, for the line grammars): a position attends
    only to earlier positions of its own line, the token itself counting as the end of the
    line it closes. `None` leaves attention plainly causal."""


class DataConfig(BaseModel, validate_assignment=True):
    batch_size: PositiveInt
    """Batch size per iteration"""

    oversample: PositiveFloat
    """Increase the number of training samples per epoch by this factor"""

    train_split: ZeroToOne
    """Fraction of data to use for training"""

    padding_chance: ZeroToOne
    """Chance of padding the beginning of a sequence with zeros"""


class TokenizerConfig(BaseModel, validate_assignment=True):
    vocabulary: list[str]
    """Unordered list of distinct tokens in the vocabulary"""

    @property
    def vocab_size(self) -> int:
        """Number of distinct tokens in the vocabulary"""
        return len(self.vocabulary)


class DatasetMetadata(BaseModel, validate_assignment=True):
    title: str

    author: str | None = None

    url: str | None = None
    """Where the dataset was downloaded from"""

    fixes: list[ExplanationStep]
    """List of fixes applied to the dataset"""

    total_chars: NonNegativeInt
    """Total number of characters in the dataset"""

    language: str | None = None
    """Language of the dataset"""


class CorpusMetadata(BaseModel, validate_assignment=True):
    tokenizer_config: TokenizerConfig
    """The tokenizer configuration used to encode the corpus"""

    total_tokens: NonNegativeInt
    """Total number of tokens in the corpus"""

    total_chars: NonNegativeInt
    """Total number of characters in the corpus"""

    sources: list[DatasetMetadata]
    """List of sources for the corpus"""


class OptimizerConfig(BaseModel, validate_assignment=True):
    weight_decay: ZeroToOne
    """Weight decay rate"""

    learning_rate: ZeroToOne
    """Learning rate"""

    betas: tuple[ZeroToOne, ZeroToOne]
    """Betas for the Adam optimizer"""


class SchedulerConfig(BaseModel, validate_assignment=True):
    epochs: PositiveInt
    """Number of epochs to train for"""

    warmup_epochs: NonNegativeFloat
    """Number of epochs to reach max learning rate"""

    min_lr_factor: ZeroToOne
    """Minimum learning rate as factor of the nominal learning rate"""


class TrainingConfig(BaseModel, validate_assignment=True):
    model: ModelConfig
    tokenizer: TokenizerConfig
    data: DataConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig

    seed: NonNegativeInt = 0
    """Seed for the PRNG keys used in model init and batch sampling"""
