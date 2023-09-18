from dataclasses import dataclass
from enum import Enum
from typing import List, Set, Tuple

import numpy as np
import pyarrow as pa
from vaex import DataFrame

# Defines the format schema for storing top_logprobs as a
# pyarrow List of List of Tuples
TOP_LOGPROBS_SCHEMA = pa.list_(pa.map_(pa.string(), pa.float32()))
TOP_K = 5


class Seq2SeqInputCols(str, Enum):
    id = "id"
    text = "text"
    input = "input"  # text is renamed to input for S2S
    label = "label"
    target = "target"  # label is renamed to target for S2S
    generated_output = "generated_output"
    split_ = "split"
    tokenized_label = "tokenized_label"
    token_label_positions = "token_label_positions"
    token_label_offsets = "token_label_offsets"

    @classmethod
    def set_cols(cls, df: DataFrame) -> DataFrame:
        """Sets the input and target columns for the dataframe"""
        return cls.set_target(cls.set_input(df))

    @classmethod
    def set_input(cls, df: DataFrame) -> DataFrame:
        """Sets the input column for the dataframe"""
        if cls.text.value in df.get_column_names():
            df = df.rename(cls.text.value, cls.input.value)

        return df

    @classmethod
    def set_target(cls, df: DataFrame) -> DataFrame:
        """Sets the target output column for the dataframe"""
        if cls.label.value in df.get_column_names():
            df = df.rename(cls.label.value, cls.target.value)

        return df


class Seq2SeqOutputCols(str, Enum):
    id = "id"
    perplexity = "perplexity"
    token_logprobs = "token_logprobs"
    top_logprobs = "top_logprobs"
    # Mypy complained about split as an attribute, so we use `split_`
    split_ = "split"
    epoch = "epoch"
    inference_name = "inference_name"


@dataclass
class AlignedTokenData:
    token_label_offsets: List[List[Tuple[int, int]]]
    token_label_positions: List[List[Set[int]]]


@dataclass
class LogprobData:
    """Data type for the top_logprobs for a single sample

    Parameters:
    -----------
    token_logprobs: np.ndarray of shape - [seq_len]
        Token label logprobs for a single sample
    top_logprobs: List[List[Tuple[str, float]]]
        List of top-k (str) predictions + corresponding logprobs
    """

    token_logprobs: np.ndarray
    top_logprobs: List[List[Tuple[str, float]]]
