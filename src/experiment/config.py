from ftfy import ExplanationStep
from pydantic import BaseModel, NonNegativeFloat, NonNegativeInt, PositiveFloat, PositiveInt

from utils.param_types import IntX8, IntX32, IntX64, ZeroToOne


class ModelConfig(BaseModel, validate_assignment=True):
    vocab_size: IntX64
    """Vocabulary size"""

    block_size: IntX64
    """Maximum sequence length"""

    n_embd: IntX8
    """Embedding dimension"""

    n_head: IntX8
    """Number of attention heads per layer"""

    n_head_dim: IntX8
    """QKV dimension per-head, usually n_embd // n_head"""

    n_ff: IntX32
    """MLP dimensions, usually 4 * n_embd"""

    n_layer: PositiveInt
    """Number of transformer blocks; also sets the residual step size (1/n_layer)"""

    residual_alpha_exp: PositiveFloat = 1.0
    """Residual step exponent: alpha = n_layer ** -exp. 1.0 = 1/n_layer, 0.5 = 1/sqrt(n_layer)."""

    normalize_sublayer: bool = True
    """Combine the residual as the nGPT LERP toward the *normalized* sublayer output
    (h + alpha*(norm(sublayer) - h)). This is the correct behaviour: it makes alpha the
    true interpolation fraction, so the per-layer step is independent of the sublayer's
    output norm (which scales like sqrt(n_embd) for the MLP). Set False for the earlier,
    incorrect additive step (h + alpha*sublayer) — kept only to reproduce that failure."""

    learnable_alpha: bool = False
    """Make the residual step a learnable scalar gain (init n_layer ** -exp) per sublayer,
    rather than a fixed constant."""


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
