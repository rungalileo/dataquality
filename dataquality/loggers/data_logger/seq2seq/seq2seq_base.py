from typing import TYPE_CHECKING, Any, List, Optional, Set, Tuple, Union, cast

import pandas as pd
import pyarrow as pa
import vaex
from vaex import DataFrame

from dataquality.exceptions import GalileoException
from dataquality.loggers.data_logger.base_data_logger import (
    ITER_CHUNK_SIZE,
    BaseGalileoDataLogger,
    DataSet,
    MetasType,
)
from dataquality.loggers.data_logger.seq2seq.formatters import (
    BaseSeq2SeqDataFormatter,
    get_data_formatter,
)
from dataquality.loggers.logger_config.seq2seq.seq2seq_base import (
    Seq2SeqLoggerConfig,
    seq2seq_logger_config,
)
from dataquality.schemas.dataframe import BaseLoggerDataFrames
from dataquality.schemas.seq2seq import Seq2SeqInputCols as S2SIC
from dataquality.schemas.seq2seq import Seq2SeqInputTempCols as S2SITC
from dataquality.schemas.seq2seq import Seq2SeqModelType
from dataquality.schemas.split import Split
from dataquality.utils.emb import convert_pa_to_np
from dataquality.utils.seq2seq.generation import add_generated_output_to_df
from dataquality.utils.seq2seq.offsets import add_target_cutoff_to_df
from dataquality.utils.vaex import rename_df

if TYPE_CHECKING:
    from datasets import Dataset


class Seq2SeqDataLogger(BaseGalileoDataLogger):
    """Seq2Seq base data logger

    This class defines the base functionality for logging input data in Seq2Seq
    tasks - i.e. shared between EncoderDecoder and DecoderOnly architectures.

    At its core, Seq2Seq data logging expects the user's tokenizer (logged through
    the provided 'watch' integration) and expects the dataset to be formatted
    as a two column datasets - corresponding to Inputs and Targets.

    During processing, we use the tokenizer to tokenize the Target data (used later
    during model output logging) and prepare for the alignment of token-level and
    string character level information.

    After processing, the following key information is extracted:
        - ids
        - texts: corresponding to the <Input> data column
        - labels: corresponding to the <Target> data column
        - token_label_offsets + token_label_positions: used for alignment of
        token level and string character level information within the UI. Note
        this only applies to the <Target> data.

    Additionally, we critically save the tokenized Target data as the ground truth
    "labels" for model output logging.

    While much of the general Seq2Seq logic can be shared between EncoderDecoder and
    DecoderOnly models, there are nuances and specific information that differentiate
    them. Therefore, the following abstract functions must be overridden by subclasses
        - validate_and_format
        - calculate_cutoffs

    Note that some shared functionality is implemented here - generally around error
    handling.
    """

    __logger_name__ = "seq2seq"
    logger_config: Seq2SeqLoggerConfig = seq2seq_logger_config
    DATA_FOLDER_EXTENSION = {"emb": "hdf5", "prob": "hdf5", "data": "arrow"}

    def __init__(self, meta: Optional[MetasType] = None) -> None:
        super().__init__(meta)
        # The target tokens (as strings) coming out of the tokenizer
        self.token_label_str: List[List[str]] = []
        # Character offsets for each token in the target indicating at each character
        # position each token starts and ends
        self.token_label_offsets: List[List[Tuple[int, int]]] = []
        # Index indicating the target tokens' position in the text (for every offset)
        self.token_label_positions: List[List[Set[int]]] = []

        self.ids: List[int] = []
        self.texts: List[str] = []
        self.labels: List[str] = []
        # Only required for Decoder-Only models
        self.formatted_prompts: List[str] = []
        # Formatter distinguishes behavior between EncoderDecoder and DecoderOnly

        self.formatter: Optional[BaseSeq2SeqDataFormatter] = None
        if self.logger_config.model_type is not None:
            self.formatter = get_data_formatter(
                self.logger_config.model_type, self.logger_config
            )

    @property
    def split_key(self) -> str:
        if self.split == Split.inference and self.inference_name is not None:
            return self.inference_name
        return str(self.split)

    def validate_and_format(self) -> None:
        """Seq2Seq validation

        Validates input lengths and existence of a tokenizer

        Further validation is done in the `formatter` for model specific
        validation (Encoder-Decoder vs Decoder-Only)
        """
        super().validate_and_format()
        label_len = len(self.labels)
        text_len = len(self.texts)
        id_len = len(self.ids)
        if label_len > 0:  # Encoder-Decoder case
            assert id_len == text_len == label_len, (
                "IDs, texts, and labels must be the same length, got "
                f"({id_len} ids, {text_len} texts, {label_len} labels)"
            )
        else:  # Decoder-Only case
            assert id_len == text_len, (
                "IDs and texts must be the same length, got "
                f"({id_len} ids, {text_len} texts)"
            )
        assert self.logger_config.tokenizer, (
            "You must set your tokenizer before logging. "
            "Use `dq.integrations.seq2seq.core.set_tokenizer`"
        )
        model_type = self.logger_config.model_type
        if self.formatter is None or model_type is None:
            raise GalileoException(
                "You must set your model type before logging. Use "
                "`dataquality.integrations.seq2seq.core.watch`"
            )

        if self.logger_config.model_type == Seq2SeqModelType.decoder_only:
            texts = self.formatted_prompts
            max_tokens = self.logger_config.max_input_tokens
        else:
            texts = self.labels
            max_tokens = self.logger_config.max_target_tokens
        assert max_tokens

        (
            batch_aligned_token_data,
            token_label_str,
            targets,
        ) = self.formatter.format_text(
            text=texts,
            ids=self.ids,
            tokenizer=self.logger_config.tokenizer,
            max_tokens=max_tokens,
            split_key=self.split_key,
        )
        self.token_label_offsets = batch_aligned_token_data.token_label_offsets
        self.token_label_positions = batch_aligned_token_data.token_label_positions
        self.token_label_str = token_label_str
        if len(targets) > 0:  # For Decoder-Only we update the 'targets' here
            self.labels = targets

    def _get_input_df(self) -> DataFrame:
        df_dict = {
            S2SIC.id.value: self.ids,
            S2SIC.input.value: self.texts,
            S2SIC.target.value: self.labels,
            S2SIC.split_.value: [self.split] * len(self.ids),
            S2SIC.token_label_positions.value: pa.array(self.token_label_positions),
            S2SIC.token_label_offsets.value: pa.array(self.token_label_offsets),
            S2SIC.token_label_str.value: pa.array(self.token_label_str),
            **self.meta,
        }
        if len(self.formatted_prompts) != 0:
            df_dict[S2SITC.formatted_prompts.value] = self.formatted_prompts

        data = vaex.from_dict(df_dict)

        if S2SIC.system_prompts in self.meta:
            # We must store nested dicts as pyarrow arrays to support vaex export
            data[S2SIC.system_prompts.value] = pa.array(self.meta[S2SIC.system_prompts])
        return data

    def _log_df(
        self,
        df: Union[pd.DataFrame, DataFrame],
        meta: Union[List[str], List[int], None] = None,
    ) -> None:
        """Helper to log a pandas or vaex df"""
        assert S2SIC.input in df, (
            f"Input column {S2SIC.input} not found in dataframe. "
            f"Please add a column titled {S2SIC.input} to your dataframe "
        )
        assert S2SIC.id in df, (
            f"ID column {S2SIC.id} not found in dataframe. "
            f"Please add a column titled {S2SIC.id} to your dataframe "
        )
        self.texts = df[S2SIC.input].tolist()
        self.ids = df[S2SIC.id].tolist()
        # Inference case
        if S2SIC.target in df:
            self.labels = df[S2SIC.target].tolist()
        if S2SITC.formatted_prompts in df:
            self.formatted_prompts = df[S2SITC.formatted_prompts].tolist()
        for meta_col in meta or []:
            self.meta[str(meta_col)] = df[meta_col].tolist()
        self.log()

    def log_dataset(
        self,
        dataset: DataSet,
        *,
        batch_size: int = ITER_CHUNK_SIZE,
        text: Union[str, int] = "input",
        id: Union[str, int] = "id",
        label: Optional[Union[str, int]] = "target",
        formatted_prompt: Union[str, int] = "formatted_label",
        split: Optional[Split] = None,
        inference_name: Optional[str] = None,
        meta: Union[List[str], List[int], None] = None,
        **kwargs: Any,
    ) -> None:
        self.validate_kwargs(kwargs)
        self.split = split
        self.inference_name = inference_name
        column_map = {text: "input", id: "id"}
        label = None if split == Split.inference else label
        if label:
            column_map[label] = "target"
        if isinstance(dataset, pd.DataFrame):
            if formatted_prompt and formatted_prompt in dataset.columns:
                column_map[formatted_prompt] = S2SITC.formatted_prompts
            dataset = dataset.rename(columns=column_map)
            self._log_df(dataset, meta)
        elif isinstance(dataset, DataFrame):
            if formatted_prompt and formatted_prompt in dataset.get_column_names():
                column_map[formatted_prompt] = S2SITC.formatted_prompts
            for chunk in range(0, len(dataset), batch_size):
                chunk_df = dataset[chunk : chunk + batch_size]
                chunk_df = rename_df(chunk_df, column_map)
                self._log_df(chunk_df, meta)
        elif self.is_hf_dataset(dataset):
            ds = cast("Dataset", dataset)  # For typing
            if formatted_prompt and formatted_prompt in ds.column_names:
                column_map[formatted_prompt] = S2SITC.formatted_prompts
            for chunk in range(0, len(ds), batch_size):
                chunk = ds[chunk : chunk + batch_size]
                chunk_df = pd.DataFrame(chunk)
                chunk_df = chunk_df.rename(columns=column_map)
                self._log_df(chunk_df, meta)
        # TODO: Maybe come back to support iterables (like tensors etc)
        else:
            raise GalileoException(
                f"Dataset must be one of pandas, vaex, or 🤗 dataset "
                f"but got {type(dataset)}"
            )

    @staticmethod
    def get_valid_attributes() -> List[str]:
        """
        Returns a list of valid attributes that for this Logger class
        :return: List[str]
        """
        return list(map(lambda x: x.value, S2SIC))

    @classmethod
    def _get_prob_cols(cls) -> List[str]:
        return ["id"]

    def create_in_out_frames(
        self,
        in_frame: DataFrame,
        dir_name: str,
        prob_only: bool,
        split: str,
        epoch_or_inf: Union[str, int],
    ) -> BaseLoggerDataFrames:
        """Formats the input data and model output data
        For Seq2Seq we need to
        - add the generated output to the input dataframe
        - calculate the text cutoffs for the input dataframe
        - call the super method to create the dataframe
        """
        # Note that we sometimes tokenize the input twice in the below methods, once for
        # finding the cutoff point of the input string used during training, and once
        # for generating
        # TODO: see if it's worth only tokenizing it once and storing it (can be large)
        in_frame = self.add_generated_output_to_df(in_frame, split)
        in_frame = self.calculate_cutoffs(in_frame)

        return super().create_in_out_frames(
            in_frame, dir_name, prob_only, split, epoch_or_inf
        )

    def convert_large_string(self, df: DataFrame) -> DataFrame:
        """Cast regular string to large_string for the text columns

        In Seq2Seq the text columns are the input and target columns.
        See BaseDataLogger.convert_large_string for more details
        """
        df_copy = df.copy()
        for text_col in [
            S2SIC.input.value,
            S2SIC.target.value,
            S2SITC.formatted_prompts.value,
        ]:
            if text_col in df_copy.get_column_names():
                # Characters are each 1 byte. If more bytes > max,
                # it needs to be large_string
                text_bytes = df_copy[text_col].str.len().sum()
                if text_bytes > self.STRING_MAX_SIZE_B:
                    df_copy[text_col] = df_copy[f'astype({text_col}, "large_string")']

            return df_copy

    def add_generated_output_to_df(
        self, df: DataFrame, split: str
    ) -> Optional[DataFrame]:
        """Adds the generated output to the dataframe
        Adds the generated output to the dataframe, and also adds the
        `token_label_positions` column
        """
        if split not in self.logger_config.generation_splits:
            return df

        model = self.logger_config.model
        tokenizer = self.logger_config.tokenizer
        max_input_tokens = self.logger_config.max_input_tokens
        generation_config = self.logger_config.generation_config
        if model is None and generation_config is not None:
            raise GalileoException(
                "To perform generation you must set your model before logging. "
                "Use `dataquality.integrations.seq2seq.core.watch`"
            )
        if tokenizer is None:
            raise GalileoException(
                "You must set your tokenizer before logging. Use "
                "`dataquality.integrations.seq2seq.core.watch`"
            )
        assert isinstance(max_input_tokens, int)

        assert generation_config is not None
        # TODO When would this be None?
        assert self.formatter is not None

        print(f"Generating {len(df)} samples for split {split}")
        # Need to specify the column to generate over!
        generation_column = S2SIC.target.value
        if self.logger_config.model_type == Seq2SeqModelType.decoder_only:
            generation_column = S2SITC.formatted_prompts.value

        df = add_generated_output_to_df(
            df,
            generation_column=generation_column,
            formatter=self.formatter,
            tokenizer=tokenizer,
            model=model,
            max_input_tokens=max_input_tokens,
            generation_config=generation_config,
            split_key=split,
        )

        return df

    @classmethod
    def separate_dataframe(
        cls, df: DataFrame, prob_only: bool = True, split: Optional[str] = None
    ) -> BaseLoggerDataFrames:
        """Separates the singular dataframe into its 3 components

        Gets the probability df, the embedding df, and the "data" df containing
        all other columns
        """
        df_copy = df.copy()
        # Separate out embeddings and probabilities into their own files
        prob_cols = cls._get_prob_cols()
        prob = df_copy[prob_cols]

        if prob_only:  # In this case, we don't care about the other columns
            emb_cols = ["id"]
            other_cols = ["id"]
        else:
            emb_cols = ["id", "emb", "x", "y", "emb_pca"]
            emb_cols = [c for c in emb_cols if c in df_copy.get_column_names()]
            ignore_cols = ["split_id"] + prob_cols + emb_cols
            other_cols = [i for i in df_copy.get_column_names() if i not in ignore_cols]
            other_cols += ["id"]

        if cls.logger_config.remove_embs:
            emb_cols = ["id"]

        emb = df_copy[emb_cols]
        if "emb" in emb.get_column_names():
            # Convert emb to numpy array
            emb = convert_pa_to_np(emb, "emb")

        return BaseLoggerDataFrames(prob=prob, emb=emb, data=df_copy)

    def calculate_cutoffs(self, df: DataFrame) -> DataFrame:
        """Calculates cuttoff indexes for the input and/or target string.

        Transformer models (or sub-modules) are trained over a maximum number of
        tokens / sequence length. This max_length controls the maximum number of
        tokens that the transformer model can process / "see." During training,
        the tokenizer uses this max_length to truncate additional tokens - so any
        tokens beyond the max token length are fully ignored.

        `calculate_cutoffs` adds relevant max_length information at the string
        character level for the `target` and/or `input` columns. This character
        info communicates to the UI how much of the respective string gets "seen"
        during processing by the model.

        In this abstract definition, we include basic error checking and compute
        the cutoffs for the target column. This logic is shared by EncoderDecoder
        and DecoderOnly models - it relies on the saved offset mapping.

        Therefore, this function adds the following columns to df:
          - 'target_cutoff': the position of the last character in the target

        See formatters (EncoderDecoder and DecoderOnly) for model specific details
        when computing `input_cutoff`.
        """
        tokenizer = self.logger_config.tokenizer
        if tokenizer is None:
            raise GalileoException(
                "You must set your tokenizer before calling dq.finish. Use "
                "`dataquality.integrations.seq2seq.core.watch`"
            )
        if self.formatter is None:
            raise GalileoException(
                "You must set your model type before logging. Use "
                "`dataquality.integrations.seq2seq.core.watch`"
            )

        # Use the computed offsets from `validate_and_format`
        target_offsets_colname = S2SIC.token_label_offsets
        if target_offsets_colname in df.get_column_names():
            df = add_target_cutoff_to_df(df, target_offsets_colname)

        # Formatter will have already been set in `validate_and_format`
        df = self.formatter.set_input_cutoff(df)

        return df
