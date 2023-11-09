from typing import List, Optional, Union
from warnings import warn

from transformers import GenerationConfig, PreTrainedModel, PreTrainedTokenizerFast

import dataquality
from dataquality.loggers.logger_config.seq2seq.decoder_only import DecoderOnlyLoggerConfig
from dataquality.loggers.logger_config.seq2seq.seq2seq_base import Seq2SeqLoggerConfig
from dataquality.schemas.split import Split
from dataquality.schemas.task_type import TaskType
from dataquality.utils.helpers import check_noop
from dataquality.utils.task_helpers import get_task_type


@check_noop
def set_tokenizer(
    tokenizer: PreTrainedTokenizerFast,
    max_input_tokens: Optional[int] = None,
    max_target_tokens: Optional[int] = None,
) -> None:
    # TODO update comment
    """Seq2seq only. Set the tokenizer for your run

    Must be a fast tokenizer, and must support `decode`, `encode`, `encode_plus`.

    We will use this tokenizer for both the input and the target. They will both be
    truncated after a certain length, which is set in the args max_input_tokens and
    max_target_tokens.

    1. tokenizer: This must be an instance of PreTrainedTokenizerFast from huggingface
        (ie T5TokenizerFast or GPT2TokenizerFast, etc). Your tokenizer should have an
        `.is_fast` property that returns True if it's a fast tokenizer.
        This class must implement the `encode`, `decode`, and `encode_plus` methods

        You can set your tokenizer via the `set_tokenizer(tok)` function imported
        from `dataquality.integrations.seq2seq.hf`

    NOTE: We assume that the tokenizer you provide is the same tokenizer used for
    training. This must be true in order to align inputs and outputs correctly. Ensure
    all necessary properties (like `add_eos_token`) are set before setting your
    tokenizer so as to match the tokenization process to your training process.
    """
    assert isinstance(
        tokenizer, PreTrainedTokenizerFast
    ), "Tokenizer must be an instance of PreTrainedTokenizerFast"
    assert getattr(tokenizer, "is_fast", False), "Tokenizer must be a fast tokenizer"
    for attr in ["encode", "decode", "encode_plus", "padding_side"]:
        assert hasattr(tokenizer, attr), f"Tokenizer must support `{attr}`"

    logger_config = dataquality.get_data_logger().logger_config
    assert isinstance(logger_config, Seq2SeqLoggerConfig)
    logger_config.tokenizer = tokenizer

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

    # This is relevant only for Encoder Decoder Models
    current_task_type = get_task_type()
    if current_task_type == TaskType.encoder_decoder:
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
    response_template: Optional[Union[str, List[int]]] = None
) -> None:
    """Seq2seq only. Log model generations for your run

    TODO Add response template to comment

    Iterates over a given dataset and logs the generations for each sample.
    `model` must be an instance of transformers PreTrainedModel and have a `generate`
    method.

    Unlike other watch functions, in this one we are just registering the model
    and generation config and not attaching any hooks to the model. We call it 'watch'
    for consistency.
    """
    logger_config = dataquality.get_data_logger().logger_config
    assert isinstance(logger_config, Seq2SeqLoggerConfig)

    assert isinstance(
        model, PreTrainedModel
    ), "model must be an instance of transformers PreTrainedModel"
    assert model.can_generate(), "model must contain a `generate` method for seq2seq"

    set_tokenizer(tokenizer, max_input_tokens, max_target_tokens)

    # Set the response template for DecoderOnly models
    if response_template:
        current_task_type = get_task_type()
        # TODO Change to decoder only
        if current_task_type == TaskType.seq2seq:
            assert isinstance(logger_config, DecoderOnlyLoggerConfig)
            logger_config.response_template = response_template
        else:
            warn(
                "The resonse_template is only used for DecoderOnly models"
            )

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
