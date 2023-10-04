from dataclasses import dataclass, field
from enum import Enum
from typing import List, Set, Tuple

import numpy as np
import pyarrow as pa
from vaex import DataFrame

# Defines the format schema for storing top_logprobs as a
# pyarrow List of List of Tuples
TOP_LOGPROBS_SCHEMA = pa.list_(pa.map_(pa.string(), pa.float32()))
TOP_K = 5

GENERATION_BATCH_SIZE = 100


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
    input_cutoff = "input_cutoff"
    target_cutoff = "target_cutoff"

    @classmethod
    def set_cols(cls, df: DataFrame) -> DataFrame:
        """Sets the input and target columns for the dataframe"""
        return cls.set_target(cls.set_input(df))

    @classmethod
    def set_input(cls, df: DataFrame) -> DataFrame:
        """Sets the input column for the dataframe"""
        if cls.text.value in df.get_column_names():
            df.rename(cls.text.value, cls.input.value)

        return df

    @classmethod
    def set_target(cls, df: DataFrame) -> DataFrame:
        """Sets the target output column for the dataframe"""
        if cls.label.value in df.get_column_names():
            df.rename(cls.label.value, cls.target.value)

        return df


class Seq2SeqOutputCols(str, Enum):
    id = "id"
    token_logprobs = "token_logprobs"
    top_logprobs = "top_logprobs"
    # Columns associated with generated output
    generated_output = "generated_output"
    generated_token_label_positions = "generated_token_label_positions"
    generated_token_label_offsets = "generated_token_label_offsets"
    generated_token_logprobs = "generated_token_logprobs"
    generated_top_logprobs = "generated_top_logprobs"
    # Mypy complained about split as an attribute, so we use `split_`
    split_ = "split"
    epoch = "epoch"
    inference_name = "inference_name"
    # Temporary columns that aren't saved to DF
    generation_data = "_generation_data"

    @staticmethod
    def generated_cols() -> List[str]:
        return [
            Seq2SeqOutputCols.generated_output.value,
            Seq2SeqOutputCols.generated_token_label_positions.value,
            Seq2SeqOutputCols.generated_token_label_offsets.value,
            Seq2SeqOutputCols.generated_token_logprobs.value,
            Seq2SeqOutputCols.generated_top_logprobs.value,
        ]


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


@dataclass
class ModelGeneration:
    generated_ids: np.ndarray
    generated_logprob_data: LogprobData


@dataclass
class BatchGenerationData:
    """Dataclass for Generated Output Data

    Stores the processed information from generated over a batch OR df
    of text Inputs. Each parameter is a List of sample data with length
    equal to the numer of samples currently in the BatchGenerationData
    object.

    Parameters:
    -----------
    generated_outputs: List[str]
        The actual generated strings for each Input sample
    generated_token_label_positions: List[List[Set[int]]]
        Token label positions for each sample
    generated_token_label_offsets: List[List[Tuple[int, int]]]
        Token label positions for each sample
    generated_token_logprobs: np.ndarray of shape - [seq_len]
        Token label logprobs for each sample
    generated_top_logprobs: List[List[List[Tuple[str, float]]]]
        top_logprobs for each sample
    """

    generated_outputs: List[str] = field(default_factory=list)
    generated_token_label_positions: List[List[Set[int]]] = field(default_factory=list)
    generated_token_label_offsets: List[List[Tuple[int, int]]] = field(
        default_factory=list
    )
    generated_token_logprobs: List[np.ndarray] = field(default_factory=list)
    generated_top_logprobs: List[List[List[Tuple[str, float]]]] = field(
        default_factory=list
    )

    def extend_from(self, batch_data: "BatchGenerationData") -> None:
        """Extend generation data from a new Batch

        Note that we favor in-place combining of batches for improved
        memory and performance.
        """
        self.generated_outputs.extend(batch_data.generated_outputs)
        self.generated_token_label_positions.extend(
            batch_data.generated_token_label_positions
        )
        self.generated_token_label_offsets.extend(
            batch_data.generated_token_label_offsets
        )
        self.generated_token_logprobs.extend(batch_data.generated_token_logprobs)
        self.generated_top_logprobs.extend(batch_data.generated_top_logprobs)
