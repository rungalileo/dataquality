from transformers import GenerationConfig, PreTrainedModel, PreTrainedTokenizerFast

from dataquality.loggers.logger_config.seq2seq import seq2seq_logger_config
from dataquality.schemas.split import Split
from dataquality.schemas.task_type import TaskType
from dataquality.utils.helpers import check_noop
from dataquality.utils.task_helpers import get_task_type


@check_noop
def set_tokenizer(tokenizer: PreTrainedTokenizerFast) -> None:
    """Seq2seq only. Set the tokenizer for your run

    Must be a fast tokenizer, and must support `decode`, `encode`, `encode_plus`
    """
    task_type = get_task_type()
    assert task_type == TaskType.seq2seq, "This method is only supported for seq2seq"
    assert isinstance(
        tokenizer, PreTrainedTokenizerFast
    ), "Tokenizer must be an instance of PreTrainedTokenizerFast"
    assert getattr(tokenizer, "is_fast", False), "Tokenizer must be a fast tokenizer"
    for attr in ["encode", "decode", "encode_plus", "padding_side"]:
        assert hasattr(tokenizer, attr), f"Tokenizer must support `{attr}`"
    seq2seq_logger_config.tokenizer = tokenizer
    # Seq2Seq doesn't have labels but we need to set this to avoid validation errors
    seq2seq_logger_config.labels = []


@check_noop
def watch(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerFast,
    generation_config: GenerationConfig,
    generate_training_data: bool = False,
) -> None:
    """Seq2seq only. Log model generations for your run

    Iterates over a given dataset and logs the generations for each sample.
    `model` must be an instance of transformers PreTrainedModel and have a `generate`
    method.

    Unlike other watch functions, in this one we are just registering the model
    and generation config and not attaching any hooks to the model. We call it 'watch'
    for consistency.
    """
    task_type = get_task_type()
    assert task_type == TaskType.seq2seq, "This method is only supported for seq2seq"
    assert isinstance(
        model, PreTrainedModel
    ), "model must be an instance of transformers PreTrainedModel"
    assert model.can_generate(), "model must contain a `generate` method for seq2seq"

    set_tokenizer(tokenizer)

    seq2seq_logger_config.model = model
    seq2seq_logger_config.generation_config = generation_config

    generation_splits = {Split.validation, Split.test}
    if generate_training_data:
        generation_splits.add(Split.training)

    seq2seq_logger_config.generation_splits = generation_splits
