from typing import List, Optional, Union
from warnings import warn

from transformers import GenerationConfig, PreTrainedModel, PreTrainedTokenizerFast

from dataquality.exceptions import GalileoException
from dataquality.loggers.logger_config.seq2seq.encoder_decoder import (
    EncoderDecoderLoggerConfig,
    encoder_decoder_logger_config,
)
from dataquality.schemas.split import Split
from dataquality.schemas.task_type import TaskType
from dataquality.utils.helpers import check_noop
from dataquality.utils.task_helpers import get_task_type


# TODO Sync with Elliott on how to differentiate between the
#  encoder_decoder vs. decoder_only logger_configs in `watch`
def _get_seg2seg_logger_config(
    task_type: TaskType,
) -> Union[EncoderDecoderLoggerConfig]:
    """Get the correct Seq2Seq logger_config based on the task_type.

    Choices between:
        1. EncoderDecoder: task_type.decoder_only
        2. DecoderOnly: task_type.decoder_only

    Raises an exception if the user has set / is using an incorrect task_type
    """
    if task_type == task_type.seq2seq:  # TODO Change to encoder_decoder
        return encoder_decoder_logger_config

    # TODO Change to encoder_decoder
    raise GalileoException(
        "Galileo's seq2seq watch method is only supported for seq2seq"
    )


@check_noop
def set_tokenizer(
    tokenizer: PreTrainedTokenizerFast,
    logger_config: Union[EncoderDecoderLoggerConfig],
    max_input_tokens: Optional[int] = None,
    max_target_tokens: Optional[int] = None,
) -> None:
    """Seq2seq only. Set the tokenizer for your run

    Must be a fast tokenizer, and must support `decode`, `encode`, `encode_plus`.

    We will use this tokenizer for both the input and the target. They will both be
    truncated after a certain length, which is set in the args max_input_tokens and
    max_target_tokens.
    """
    assert isinstance(
        tokenizer, PreTrainedTokenizerFast
    ), "Tokenizer must be an instance of PreTrainedTokenizerFast"
    assert getattr(tokenizer, "is_fast", False), "Tokenizer must be a fast tokenizer"
    for attr in ["encode", "decode", "encode_plus", "padding_side"]:
        assert hasattr(tokenizer, attr), f"Tokenizer must support `{attr}`"
    logger_config.tokenizer = tokenizer

    # This is relevant only for Encoder Decoder Models
    logger_config.max_input_tokens = max_input_tokens
    if logger_config.max_input_tokens is None:
        logger_config.max_input_tokens = tokenizer.model_max_length
        warn(
            (
                "The argument max_input_tokens is not set, we will use the value "
                f"{tokenizer.model_max_length} from tokenizer.model_max_length. If you "
                "tokenized the input with another value, this can lead to confusing "
                "insights about this training run."
            )
        )

    if type(logger_config) == EncoderDecoderLoggerConfig:
        logger_config.max_target_tokens = max_target_tokens
        if logger_config.max_target_tokens is None:
            logger_config.max_target_tokens = tokenizer.model_max_length
            warn(
                (
                    "The argument max_target_tokens is not set, we will use the value "
                    f"{tokenizer.model_max_length} from tokenizer.model_max_length. "
                    f"If you tokenized the target with another value, this can lead "
                    f"to confusing insights about this training run."
                )
            )
    else:
        warn(
            "The argument max_target_tokens is only used when working with "
            "EncoderDecoder models. This value will be ignored."
        )

    # Seq2Seq doesn't have labels but we need to set this to avoid validation errors
    logger_config.labels = []


@check_noop
def watch(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerFast,
    generation_config: GenerationConfig,
    generation_splits: Optional[List[str]] = None,
    max_input_tokens: Optional[int] = None,
    max_target_tokens: Optional[int] = None,
) -> None:
    # TODO Update comment
    """Seq2seq only. Log model generations for your run

    Iterates over a given dataset and logs the generations for each sample.
    `model` must be an instance of transformers PreTrainedModel and have a `generate`
    method.

    Unlike other watch functions, in this one we are just registering the model
    and generation config and not attaching any hooks to the model. We call it 'watch'
    for consistency.
    """
    task_type = get_task_type()
    # Get the corresponding logger config - handling error checking
    logger_config = _get_seg2seg_logger_config(task_type)
    assert isinstance(
        model, PreTrainedModel
    ), "model must be an instance of transformers PreTrainedModel"
    assert model.can_generate(), "model must contain a `generate` method for seq2seq"

    set_tokenizer(tokenizer, logger_config, max_input_tokens, max_target_tokens)

    logger_config.model = model
    logger_config.generation_config = generation_config

    generation_splits = generation_splits or []
    generation_splits_set = {Split.test}
    for split in generation_splits:
        if split not in Split.get_valid_keys():
            warn(
                f"Ignoring invalid generation split {split}, "
                f"the valid splits are {Split.get_valid_keys()}"
            )
            continue

        generation_splits_set.add(Split[split])

    logger_config.generation_splits = generation_splits_set
