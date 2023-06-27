from typing import TYPE_CHECKING, Any, List, Optional, Set, Tuple, Union, cast

import pandas as pd
import pyarrow as pa
import vaex
from vaex.dataframe import DataFrame

from dataquality.exceptions import GalileoException
from dataquality.loggers.data_logger.base_data_logger import (
    ITER_CHUNK_SIZE,
    BaseGalileoDataLogger,
    DataSet,
    MetasType,
)
from dataquality.loggers.logger_config.seq2seq import seq2seq_logger_config
from dataquality.schemas.seq2seq import Seq2SeqCols as C
from dataquality.schemas.split import Split
from dataquality.utils.seq2seq import (
    align_tokens_to_character_spans,
)
from dataquality.utils.vaex import rename_df

if TYPE_CHECKING:
    from datasets import Dataset


class Seq2SeqDataLogger(BaseGalileoDataLogger):
    """Logging input data for Seq2Seq fine-tuning tasks

    Logging input data for Seq2Seq requires 2 pieces of information:
    1. tokenizer: This must be an instance of PreTrainedTokenizerFast from huggingface
        (ie T5TokenizerFast or GPT2TokenizerFast, etc). Your tokenizer should have an
        `.is_fast` property that returns True if it's a fast tokenizer.
        This class must implement the `encode`, `decode`, and `encode_plus` methods

        You can set your tokenizer via the `dq.set_tokenizer(tok)` function
    2. A dataset (pandas/huggingface etc) with input strings and output labels and ids.
        Ex: Billsum dataset, with `text` input and `summary` as the label
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
        from datasets import load_dataset
        from transformers import T5TokenizerFast

        tokenizer = T5TokenizerFast.from_pretrained("t5-small")
        ds = load_dataset("billsum")
        # Add `id` column to each dataset split as the idx
        ds = ds.map(lambda x,idx : {"id":idx},with_indices=True)
        dq.init("seq2seq")
        dq.set_tokenizer(tokenizer)
        dq.log_dataset(ds["train"], label="summary", split="train")

    NOTE: We assume that the tokenizer you provide is the same tokenizer used for
    training. This must be true in order to align inputs and outputs correctly. Ensure
    all necessary properties (like `add_eos_token`) are set before setting your
    tokenizer so as to match the tokenization process to your training process.
    """

    __logger_name__ = "seq2seq"
    logger_config = seq2seq_logger_config

    def __init__(self, meta: Optional[MetasType] = None) -> None:
        super().__init__(meta)
        # Tokens IDs in a given input string
        self.tokenized_labels: List[List[int]] = []
        # Character offsets for each token (from tokenized_inputs) in the dataset
        self.token_label_offsets: List[List[Tuple[int, int]]] = []
        # Index (or indices) into the token array for every offset
        self.token_label_positions: List[List[Set[int]]] = []
        self.ids: List[int] = []
        self.texts: List[str] = []
        self.labels: List[str] = []

    @property
    def token_map_key(self) -> str:
        if self.split == Split.inference and self.inference_name is not None:
            return self.inference_name
        return str(self.split)

    def validate_and_format(self) -> None:
        super().validate_and_format()
        label_len = len(self.labels)
        text_len = len(self.texts)
        id_len = len(self.ids)
        assert id_len == text_len == label_len, (
            "IDs, text, and labels must be the same length, got "
            f"({id_len} ids, {text_len} text, {label_len} labels)"
        )
        assert (
            self.logger_config.tokenizer
        ), "You must set your tokenizer before logging. Use `dq.set_tokenizer`"
        encoded_data = self.logger_config.tokenizer(
            self.labels, return_offsets_mapping=True
        )
        self.tokenized_labels = encoded_data["input_ids"]
        aligned_data = align_tokens_to_character_spans(encoded_data["offset_mapping"])
        self.token_label_offsets = aligned_data.token_label_offsets
        self.token_label_positions = aligned_data.token_label_positions

        id_to_tokens = dict(zip(self.ids, self.tokenized_labels))
        self.logger_config.id_to_tokens[self.token_map_key].update(id_to_tokens)

    def _get_input_df(self) -> DataFrame:
        return vaex.from_dict(
            {
                C.id.value: self.ids,
                C.text.value: self.texts,
                C.label.value: self.labels,
                # TODO: Do we need to save this to the server?
                C.tokenized_label.value: pa.array(self.tokenized_labels),
                C.token_label_positions.value: pa.array(self.token_label_positions),
                C.token_label_offsets.value: pa.array(self.token_label_offsets),
            }
        )

    def _log_df(
        self,
        df: Union[pd.DataFrame, DataFrame],
        meta: Union[List[str], List[int], None] = None,
    ) -> None:
        """Helper to log a pandas or vaex df"""
        self.texts = df["text"].tolist()
        self.ids = df["id"].tolist()
        # Inference case
        if "label" in df:
            self.labels = df["label"].tolist()
        for meta_col in meta or []:
            self.meta[str(meta_col)] = df[meta_col].tolist()
        self.log()

    def log_dataset(
        self,
        dataset: DataSet,
        *,
        batch_size: int = ITER_CHUNK_SIZE,
        text: Union[str, int] = "text",
        id: Union[str, int] = "id",
        label: Optional[Union[str, int]] = "label",
        split: Optional[Split] = None,
        inference_name: Optional[str] = None,
        meta: Union[List[str], List[int], None] = None,
        **kwargs: Any,
    ) -> None:
        self.validate_kwargs(kwargs)
        self.split = split
        self.inference_name = inference_name
        column_map = {text: "text", id: "id"}
        label = None if split == Split.inference else label
        if label:
            column_map[label] = "label"
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
        return list(map(lambda x: x.value, C))
