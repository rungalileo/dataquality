from typing import Optional

from vaex.dataframe import DataFrame

from dataquality.loggers.data_logger.base_data_logger import (
    MetasType,
)
from dataquality.loggers.data_logger.seq2seq.seq2seq_base import Seq2SeqDataLogger
from dataquality.loggers.logger_config.seq2seq.encoder_decoder import (
    EncoderDecoderLoggerConfig,
    encoder_decoder_logger_config,
)
from dataquality.schemas.seq2seq import Seq2SeqInputCols as C
from dataquality.utils.seq2seq.offsets import (
    align_tokens_to_character_spans,
    get_cutoff_from_saved_offsets,
    get_cutoff_from_truncated_tokenization,
)


class EncoderDecoderDataLogger(Seq2SeqDataLogger):
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

    NOTE 2: Unlike EncoderOnly models, EncoderDecoder models explicitly separate the
    processing of the <Input> and <Target> data. Therefore, we do not need any
    additional information to isolate / extract information on the <Target> data.
    """

    __logger_name__ = "encoder_decoder"
    logger_config: EncoderDecoderLoggerConfig = encoder_decoder_logger_config
    DATA_FOLDER_EXTENSION = {"emb": "hdf5", "prob": "hdf5", "data": "arrow"}

    def __init__(self, meta: Optional[MetasType] = None) -> None:
        super().__init__(meta)

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

        id_to_tokens = dict(zip(self.ids, tokenized_labels))
        self.logger_config.id_to_tokens[self.token_map_key].update(id_to_tokens)

    @classmethod
    def calculate_cutoffs(cls, df: DataFrame) -> DataFrame:
        """Calculate the cutoff index for the input and target strings.


        When using Encoder-Decoder models, the input AND target tokens are truncated
        based on the respective Encoder (input) / Decoder (target) max_lengths
        OR user specified max_lengths (note: these may be different between the
        Encoder and Decoder).

        The model only "sees"/processes the tokens that remain after truncation,
        for example if max_length=512 for the Encoder, no matter how long the Input,
        the model will only process the first 512 tokens and ignore the rest.

        This function adds two columns to the df:
          - 'input_cutoff': the position of the last character in the input.
          - 'target_cutoff': the position of the last character in the target.
        """
        # Error checking
        super().calculate_cutoffs(df)

        # TODO we may be able to take advantage of shared code with Decoder
        tokenizer = cls.logger_config.tokenizer
        max_input_length = cls.logger_config.max_input_tokens
        df[C.input_cutoff.value] = get_cutoff_from_truncated_tokenization(
            df, C.text, tokenizer, max_input_length
        )

        target_offsets_colname = C.token_label_offsets
        if target_offsets_colname in df.get_column_names():
            df[C.target_cutoff.value] = get_cutoff_from_saved_offsets(
                df, target_offsets_colname
            )

        return df
