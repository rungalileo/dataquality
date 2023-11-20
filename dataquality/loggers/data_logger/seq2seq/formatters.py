from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Type

from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerFast
from vaex import DataFrame

from dataquality.loggers.data_logger.seq2seq.seq2seq_base import Seq2SeqDataLogger
from dataquality.loggers.logger_config.seq2seq.seq2seq_base import Seq2SeqLoggerConfig
from dataquality.schemas.seq2seq import Seq2SeqInputCols as S2SIC
from dataquality.schemas.seq2seq import Seq2SeqModelType
from dataquality.utils.seq2seq.decoder_only import extract_tokenized_responses
from dataquality.utils.seq2seq.offsets import (
    add_input_cutoff_to_df,
    add_target_cutoff_to_df,
    align_response_tokens_to_character_spans,
    align_tokens_to_character_spans,
)


class BaseSeq2SeqDataFormatter(ABC):
    def __init__(self, logger_config: Seq2SeqLoggerConfig) -> None:
        self.logger_config = logger_config

    @abstractmethod
    def set_input_cutoff(self, df: DataFrame) -> DataFrame:
        pass

    @abstractmethod
    def format_text(
        self,
        tokenizer: PreTrainedTokenizerFast,
        text: List[str],
        max_tokens: Optional[int],
        data_logger: Seq2SeqDataLogger,
    ) -> None:
        pass


class EncoderDecoderDataFormatter(BaseSeq2SeqDataFormatter):
    """Seq2Seq data logger for EncoderDecoder models

    Logging input data for EncoderDecoder models requires:
    1. tokenizer: This must be an instance of PreTrainedTokenizerFast from huggingface
        (ie T5TokenizerFast or GPT2TokenizerFast, etc). Your tokenizer should have an
        `.is_fast` property that returns True if it's a fast tokenizer.
        This class must implement the `encode`, `decode`, and `encode_plus` methods

        You can set your tokenizer via either the seq2seq `set_tokenizer()` or
        `watch(tokenizer, ...)` functions in `dataquality.integrations.seq2seq.core`
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
        from dataquality.integrations.seq2seq.core import set_tokenizer
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
            "encoder_decoder",
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

    def format_text(
        self,
        tokenizer: PreTrainedTokenizerFast,
        text: List[str],
        max_tokens: Optional[int],
        data_logger: Seq2SeqDataLogger,
    ) -> None:
        """Further validation for Encoder-Decoder

        For Encoder-Decoder we need to:
            - Save the offsets and positions of the tokenized labels
                Allows us to extract token level information from the full
                sample text
            - Save the target token id
                Equivalent to the "ground truth", allows us to get logprob per
                token later on

        We achieve this by:
            - Tokenize the target texts using `max_target_tokens`
            - From the tokenized outputs generate the corresponding token alignments
                (i.e. label_offsets and lable_positions).
            - Save ground-truth token id in `id_to_tokens` map, mapping
                sample id to tokenized label (sample_id -> List[token_id])
        """
        targets = text
        max_target_tokens = max_tokens
        encoded_data = tokenizer(
            targets,
            return_offsets_mapping=True,
            max_length=max_target_tokens,
            truncation=True,
        )
        tokenized_labels = encoded_data["input_ids"]
        aligned_data = align_tokens_to_character_spans(encoded_data["offset_mapping"])
        data_logger.token_label_offsets = aligned_data.token_label_offsets
        data_logger.token_label_positions = aligned_data.token_label_positions

        # Save the tokenized response labels for each samples
        id_to_tokens = dict(zip(data_logger.ids, tokenized_labels))
        self.logger_config.id_to_tokens[data_logger.split_key].update(id_to_tokens)

    def set_input_cutoff(self, df: DataFrame) -> DataFrame:
        """Calculate the cutoff index for the input strings.

        When using Encoder-Decoder models, the input tokens are truncated
        based on the respective Encoders max_lengths OR the user specified
        max_length (note: these may be different between Encoder and Decoder
        - see `max_input_tokens` vs. `max_target_tokens).

        This function adds one column to the df:
          - 'input_cutoff': the position of the last character in the input.
        """
        tokenizer = self.logger_config.tokenizer
        max_input_length = self.logger_config.max_input_tokens
        df = add_input_cutoff_to_df(df, tokenizer, S2SIC.input, max_input_length)

        return df


class DecoderOnlyDataFormatter(BaseSeq2SeqDataFormatter):
    """Seq2Seq data logger for DecoderOnly models

    Logging input data for DecoderOnly models requires:
    1. tokenizer: This must be an instance of PreTrainedTokenizerFast from huggingface
        (ie T5TokenizerFast or GPT2TokenizerFast, etc). Your tokenizer should have an
        `.is_fast` property that returns True if it's a fast tokenizer.
        This class must implement the `encode`, `decode`, and `encode_plus` methods

        You can set your tokenizer via either the seq2seq `set_tokenizer()` or
        `watch(tokenizer, ...)` functions in `dataquality.integrations.seq2seq.core`
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
        from dataquality.integrations.seq2seq.core import set_tokenizer
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
            "encoder_decoder",
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

    def format_text(
        self,
        tokenizer: PreTrainedTokenizerFast,
        text: List[str],
        max_tokens: Optional[int],
        data_logger: Seq2SeqDataLogger,
    ) -> None:
        """Further formatting for Decoder-Only

        Text is the formatted prompt of combined input/target

        Tokenize text using the user's `max_input_tokens`. From
        the tokenized outputs generate the corresponding token alignments
        (i.e. label_offsets and lable_positions).

        Save the tokenized labels for each sample as `id_to_tokens`. This
        is essential during model logging for extracting GT token label
        information.

        We also save a `formatted_prompt_lengths` map used later to remove
        padding tokens.
        """
        # For decoder-only the text is the formatted prompt (input/target combined)
        formatted_prompts = text
        max_input_tokens = max_tokens
        encoded_data = tokenizer(
            formatted_prompts,
            max_length=max_input_tokens,
            truncation=True,
        )
        # Tokenized input/target combination
        tokenized_formatted_prompts = encoded_data["input_ids"]

        # Split each sample based on the location of the response template
        # This is equivalent to `tokenized_labels` in encoder-decoder
        assert self.logger_config.response_template  # Necessary for linting
        tokenized_labels = extract_tokenized_responses(
            tokenized_formatted_prompts, self.logger_config.response_template
        )

        # Decode then re-tokenize just the response labels to get correct offsets
        for tokenized_response in tqdm(
            tokenized_labels,
            leave=False,
            desc="Aligning string characters with tokenizer representation",
        ):
            aligned_data = align_response_tokens_to_character_spans(
                tokenizer,
                tokenized_response,
                max_input_tokens,
            )
            data_logger.token_label_offsets.append(aligned_data.token_label_offsets[0])
            data_logger.token_label_positions.append(
                aligned_data.token_label_positions[0]
            )

        # Save the tokenized response labels for each samples
        id_to_tokens = dict(zip(data_logger.ids, tokenized_labels))
        self.logger_config.id_to_tokens[data_logger.split_key].update(id_to_tokens)

        # Save the length of the formatted prompt - used later to remove padding
        formatted_prompt_lengths = [
            len(prompt) for prompt in tokenized_formatted_prompts
        ]
        id_to_formatted_prompt_length = dict(
            zip(data_logger.ids, formatted_prompt_lengths)
        )
        self.logger_config.id_to_formatted_prompt_length[data_logger.split_key].update(
            id_to_formatted_prompt_length
        )

    def set_input_cutoff(self, df: DataFrame) -> DataFrame:
        """Calculate the cutoff index for the inputs

        Set the cutoff for the Input to just be the entire sample
            i.e. the length of `input`
        """
        # Assign input_cutoff always to be the full strings
        df[S2SIC.input_cutoff.value] = df[S2SIC.input].str.len()

        target_offsets_colname = S2SIC.token_label_offsets
        if target_offsets_colname in df.get_column_names():
            df = add_target_cutoff_to_df(df, target_offsets_colname)

        return df


FORMATTER_MAP: Dict[Seq2SeqModelType, Type[BaseSeq2SeqDataFormatter]] = {
    Seq2SeqModelType.encoder_decoder: EncoderDecoderDataFormatter,
    Seq2SeqModelType.decoder_only: DecoderOnlyDataFormatter,
}


def get_data_formatter(
    model_type: Seq2SeqModelType, logger_config: Seq2SeqLoggerConfig
) -> BaseSeq2SeqDataFormatter:
    """Returns the data formatter for the given model_type"""
    return FORMATTER_MAP[model_type](logger_config)
