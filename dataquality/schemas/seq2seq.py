from dataclasses import dataclass
from enum import Enum
from typing import List, Set, Tuple

from vaex import DataFrame


class Seq2SeqInputCols(str, Enum):
    id = "id"
    text = "text"
    input = "input"  # text is renamed to input for S2S
    label = "label"
    target_output = "target_output"  # label is renamed to target_output for S2S
    generated_output = "generated_output"
    split_ = "split"
    tokenized_label = "tokenized_label"
    token_label_positions = "token_label_positions"
    token_label_offsets = "token_label_offsets"

    @classmethod
    def set_cols(cls, df: DataFrame) -> DataFrame:
        """Sets the input and target_output columns for the dataframe"""
        return cls.set_target_output(cls.set_input(df))

    @classmethod
    def set_input(cls, df: DataFrame) -> DataFrame:
        """Sets the input column for the dataframe"""
        if cls.text.value in df.get_column_names():
            df = df.rename(cls.text.value, cls.input.value)

        return df

    @classmethod
    def set_target_output(cls, df: DataFrame) -> DataFrame:
        """Sets the target output column for the dataframe"""
        if cls.label.value in df.get_column_names():
            df = df.rename(cls.label.value, cls.target_output.value)

        return df


class Seq2SeqOutputCols(str, Enum):
    id = "id"
    dep = "data_error_potential"
    token_deps = "token_deps"
    token_gold_probs = "token_gold_probs"
    # Mypy complained about split as an attribute, so we use `split_`
    split_ = "split"
    epoch = "epoch"
    inference_name = "inference_name"


@dataclass
class AlignedTokenData:
    token_label_offsets: List[List[Tuple[int, int]]]
    token_label_positions: List[List[Set[int]]]
