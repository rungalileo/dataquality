from abc import ABC
from typing import Any, Dict, List, Optional, Type, Union, cast

import pandas as pd
from datasets import Dataset
from vaex import DataFrame

from dataquality.exceptions import GalileoException
from dataquality.loggers.data_logger.base_data_logger import (
    ITER_CHUNK_SIZE,
    DataSet,
)
from dataquality.loggers.logger_config.seq2seq.seq2seq_base import Seq2SeqLoggerConfig
from dataquality.schemas.seq2seq import Seq2SeqInputCols as S2SIC
from dataquality.schemas.seq2seq import Seq2SeqModelTypes
from dataquality.schemas.split import Split
from dataquality.utils.seq2seq.decoder_only import extract_tokenized_responses
from dataquality.utils.seq2seq.offsets import (
    add_input_cutoff_to_df,
    add_target_cutoff_to_df,
    align_response_tokens_to_character_spans,
    align_tokens_to_character_spans,
)
from dataquality.utils.vaex import rename_df


class BaseSeq2SeqDataFormatter(ABC):
    def __init__(self, logger_config: Seq2SeqLoggerConfig) -> None:
        self.logger_config = logger_config


class EncoderDecoderDataFormatter(BaseSeq2SeqDataFormatter):
    """Seq2Seq data logger for EncoderDecoder models

    Logging input data for EncoderDecoder models requires:
    1. tokenizer: This must be an instance of PreTrainedTokenizerFast from huggingface
        (ie T5TokenizerFast or GPT2TokenizerFast, etc). Your tokenizer should have an
        `.is_fast` property that returns True if it's a fast tokenizer.
        This class must implement the `encode`, `decode`, and `encode_plus` methods

        You can set your tokenizer via either the seq2seq `set_tokenizer()` or
        `watch(..., tokenizer, ...)` functions in `dataquality.integrations.seq2seq.hf`
    2. A two column (i.e. completion) dataset (pandas/huggingface etc) with string
        'text' (model <Input> / <Instruction> / <Prompt>, ...) and 'label' (model
        <Target> / (<Completion> / ...) columns + a data sample id column.
        Ex: Billsum dataset, with `text` <Input> and `summary` as the <Label>
        id  text	                        summary
        0	SECTION 1. LIABILITY ...	    Shields a business entity ...
        1	SECTION 1. SHORT TITLE.\n\n ...	Human Rights Information Act ...
        2	SECTION 1. SHORT TITLE.\n\n ...	Jackie Robinson Commemorative Coin ...
        3	SECTION 1. NONRECOGNITION ...	Amends the Internal Revenue Code to ...
        4	SECTION 1. SHORT TITLE.\n\n ...	Native American Energy Act - (Sec. 3...

        You can log your dataset via the `dq.log_dataset` function, passing in the
        column mapping as necessary for `text`, `label`, and `id`
        `dq.log_dataset(ds, text="text", label="summary", id="id")`

    Putting it all together:
        from dataquality.integrations.seq2seq.hf import set_tokenizer
        from datasets import load_dataset
        from transformers import T5TokenizerFast

        tokenizer = T5TokenizerFast.from_pretrained("t5-small")
        ds = load_dataset("billsum")
        # Add `id` column to each dataset split as the idx
        ds = ds.map(lambda x,idx : {"id":idx},with_indices=True)
        dq.init("seq2seq")
        # You can either use `set_tokenizer()` or `watch()`
        set_tokenizer(
            tokenizer,
            max_input_tokens=512,
            max_target_tokens=128
        )
        dq.log_dataset(ds["train"], label="summary", split="train")

    NOTE: We assume that the tokenizer you provide is the same tokenizer used for
    training. This must be true in order to align inputs and outputs correctly. Ensure
    all necessary properties (like `add_eos_token`) are set before setting your
    tokenizer as to match the tokenization process to your training process.

    NOTE 2: Unlike DecoderOnly models, EncoderDecoder models explicitly separate the
    processing of the <Input> and <Target> data. Therefore, we do not need any
    additional information to isolate / extract information on the <Target> data.
    """

    def validate_and_format(self) -> None:
        """Format Encoder-Decoder Data Format

        Tokenize self.labels, using the user's `max_taget_tokens`. From
        the tokenized outputs generate the corresponding token alignments
        (i.e. label_offsets and lable_positions).

        Save the tokenized labels for each sample as `id_to_tokens`. This
        is essential during model logging for extracting GT token label
        information.

        Note: the parent Seq2SeqDataLogger.validate_and_format() handles
        common data type validation.
        """
        super().validate_and_format()
        # We ensure tokenizer is set in the parent class
        encoded_data = self.logger_config.tokenizer(  # type: ignore
            self.labels,
            return_offsets_mapping=True,
            max_length=self.logger_config.max_target_tokens,
            truncation=True,
        )
        tokenized_labels = encoded_data["input_ids"]
        aligned_data = align_tokens_to_character_spans(encoded_data["offset_mapping"])
        self.token_label_offsets = aligned_data.token_label_offsets
        self.token_label_positions = aligned_data.token_label_positions

        # Save the target labels for each sample
        id_to_tokens = dict(zip(self.ids, tokenized_labels))
        self.logger_config.id_to_tokens[self.token_map_key].update(id_to_tokens)

    @classmethod
    def calculate_cutoffs(cls, df: DataFrame) -> DataFrame:
        """Calculate the cutoff index for the input strings.


        When using Encoder-Decoder models, the input tokens are truncated
        based on the respective Encoders max_lengths OR the user specified
        max_length (note: these may be different between Encoder and Decoder
        - see `max_input_tokens` vs. `max_target_tokens).

        This function adds one column to the df:
          - 'input_cutoff': the position of the last character in the input.
        """
        # Error checking + target_cutoff computation
        super().calculate_cutoffs(df)

        tokenizer = cls.logger_config.tokenizer
        max_input_length = cls.logger_config.max_input_tokens
        df = add_input_cutoff_to_df(df, tokenizer, S2SIC.text, max_input_length)

        return df


class DecoderOnlyDataFormatter(BaseSeq2SeqDataFormatter):
    """Seq2Seq data logger for DecoderOnly models

    TODO Update
    Logging input data for DecoderOnly models requires:
    1. tokenizer: This must be an instance of PreTrainedTokenizerFast from huggingface
        (ie T5TokenizerFast or GPT2TokenizerFast, etc). Your tokenizer should have an
        `.is_fast` property that returns True if it's a fast tokenizer.
        This class must implement the `encode`, `decode`, and `encode_plus` methods

        You can set your tokenizer via either the seq2seq `set_tokenizer()` or
        `watch(..., tokenizer, ...)` functions in `dataquality.integrations.seq2seq.hf`
    2. A two column (i.e. completion) dataset (pandas/huggingface etc) with string
        'text' (model <Input> / <Instruction> / <Prompt>, ...) and 'label' (model
        <Target> / (<Completion> / ...) columns + a data sample id column.
        Ex: Billsum dataset, with `text` <Input> and `summary` as the <Label>
        id  text	                        summary
        0	SECTION 1. LIABILITY ...	    Shields a business entity ...
        1	SECTION 1. SHORT TITLE.\n\n ...	Human Rights Information Act ...
        2	SECTION 1. SHORT TITLE.\n\n ...	Jackie Robinson Commemorative Coin ...
        3	SECTION 1. NONRECOGNITION ...	Amends the Internal Revenue Code to ...
        4	SECTION 1. SHORT TITLE.\n\n ...	Native American Energy Act - (Sec. 3...

        You can log your dataset via the `dq.log_dataset` function, passing in the
        column mapping as necessary for `text`, `label`, and `id`
        `dq.log_dataset(ds, text="text", label="summary", id="id")`

    Putting it all together:
        from dataquality.integrations.seq2seq.hf import set_tokenizer
        from datasets import load_dataset
        from transformers import T5TokenizerFast

        tokenizer = T5TokenizerFast.from_pretrained("t5-small")
        ds = load_dataset("billsum")
        # Add `id` column to each dataset split as the idx
        ds = ds.map(lambda x,idx : {"id":idx},with_indices=True)
        dq.init("seq2seq")
        # You can either use `set_tokenizer()` or `watch()`
        set_tokenizer(
            tokenizer,
            max_input_tokens=512,
            max_target_tokens=128
        )
        dq.log_dataset(ds["train"], label="summary", split="train")

    NOTE: We assume that the tokenizer you provide is the same tokenizer used for
    training. This must be true in order to align inputs and outputs correctly. Ensure
    all necessary properties (like `add_eos_token`) are set before setting your
    tokenizer as to match the tokenization process to your training process.

    NOTE 2: Unlike DecoderOnly models, EncoderDecoder models explicitly separate the
    processing of the <Input> and <Target> data. Therefore, we do not need any
    additional information to isolate / extract information on the <Target> data.
    """

    def validate_and_format(self) -> None:
        """Format Decoder-Only Data Format

        # TODO update comment!
        Format Encoder-Decoder Data Format

        Tokenize self.labels, using the user's `max_taget_tokens`. From
        the tokenized outputs generate the corresponding token alignments
        (i.e. label_offsets and lable_positions).

        Save the tokenized labels for each sample as `id_to_tokens`. This
        is essential during model logging for extracting GT token label
        information.

        Note: the parent Seq2SeqDataLogger.validate_and_format() handles
        common data type validation.
        """
        super().validate_and_format()
        encoded_data = self.logger_config.tokenizer(  # type: ignore
            self.formatted_prompts,
            max_length=self.logger_config.max_input_tokens,
            truncation=True,
        )
        tokenized_formatted_prompts = encoded_data["input_ids"]

        # Split each sample based on the location of the response template
        tokenized_responses = extract_tokenized_responses(
            tokenized_formatted_prompts, self.logger_config.response_template
        )

        # Decode then re-tokenize just the response labels to get correct offsets
        for tokenized_response in tokenized_responses:
            aligned_data = align_response_tokens_to_character_spans(
                tokenized_response,
                self.logger_config.tokenizer,
                self.logger_config.max_input_tokens,
            )

            self.token_label_offsets.append(aligned_data.token_label_offsets[0])
            self.token_label_positions.append(aligned_data.token_label_positions[0])

        # Save the tokenized response labels for each samples
        id_to_tokens = dict(zip(self.ids, tokenized_responses))
        self.logger_config.id_to_tokens[self.token_map_key].update(id_to_tokens)

        # Save the length of the formatted prompt - used later to remove padding
        formatted_prompt_lengths = [
            len(prompt) for prompt in tokenized_formatted_prompts
        ]
        id_to_formatted_prompt_length = dict(zip(self.ids, formatted_prompt_lengths))
        self.logger_config.id_to_formatted_prompt_length[self.token_map_key].update(
            id_to_formatted_prompt_length
        )

    def _log_df(
        self,
        df: Union[pd.DataFrame, DataFrame],
        meta: Union[List[str], List[int], None] = None,
    ) -> None:
        """Assigns `formatted_prompt`"""
        self.formatted_prompts = df["galileo_formatted_prompt"].tolist()
        super()._log_df(df, meta=meta)

    def log_dataset(
        self,
        dataset: DataSet,
        *,
        batch_size: int = ITER_CHUNK_SIZE,
        text: Union[str, int] = "text",
        id: Union[str, int] = "id",
        label: Optional[Union[str, int]] = "label",
        formatted_prompt: Union[str, int] = "formatted_label",
        split: Optional[Split] = None,
        inference_name: Optional[str] = None,
        meta: Union[List[str], List[int], None] = None,
        **kwargs: Any,
    ) -> None:
        """Overrides Seq2Seq base `log_dataset`

        Capture the user's `formatted_prompt` data column,
        representing the full input for Decoder-Only models.
        """
        self.validate_kwargs(kwargs)
        self.split = split
        self.inference_name = inference_name
        column_map = {
            text: "text",
            label: "label",
            formatted_prompt: "galileo_formatted_prompt",
            id: "id",
        }
        if isinstance(dataset, pd.DataFrame):
            dataset = dataset.rename(columns=column_map)
            self._log_df(dataset, meta)
        elif isinstance(dataset, DataFrame):
            for chunk in range(0, len(dataset), batch_size):
                chunk_df = dataset[chunk : chunk + batch_size]
                chunk_df = rename_df(chunk_df, column_map)
                self._log_df(chunk_df, meta)
        elif self.is_hf_dataset(dataset):
            ds = cast("Dataset", dataset)  # For typing
            for chunk in range(0, len(ds), batch_size):
                chunk = ds[chunk : chunk + batch_size]
                chunk_df = pd.DataFrame(chunk)
                chunk_df = chunk_df.rename(columns=column_map)
                self._log_df(chunk_df, meta)
        else:
            raise GalileoException(
                f"Dataset must be one of pandas, vaex, or 🤗 dataset "
                f"but got {type(dataset)}"
            )

    @classmethod
    def calculate_cutoffs(cls, df: DataFrame) -> DataFrame:
        """Calculate the cutoff index for the input and target strings.

        For now, we do the following:
            - Compute the cutoff for the Target based on the processed
            self.token_label_positions + offsets.
            - Set the cutoff for the Input to just be the entire sample
            i.e. the length of `text`
        """
        # Error checking
        super().calculate_cutoffs(df)

        # Assign input_cutoff always to be the full strings
        df[S2SIC.input_cutoff.value] = df[S2SIC.text].str.len()

        # Use the computed offsets from `validate_and_format`
        target_offsets_colname = S2SIC.token_label_offsets
        if target_offsets_colname in df.get_column_names():
            df = add_target_cutoff_to_df(df, target_offsets_colname)

        return df


FORMATTER_MAP: Dict[Seq2SeqModelTypes, Type[BaseSeq2SeqDataFormatter]] = {
    Seq2SeqModelTypes.encoder_decoder: EncoderDecoderDataFormatter,
    Seq2SeqModelTypes.decoder_only: DecoderOnlyDataFormatter,
}


def get_data_formatter(
    model_type: Seq2SeqModelTypes, logger_config: Seq2SeqLoggerConfig
) -> BaseSeq2SeqDataFormatter:
    """Returns the data formatter for the given model_type"""
    return FORMATTER_MAP[model_type](logger_config)
