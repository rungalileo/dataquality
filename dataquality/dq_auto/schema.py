from dataclasses import dataclass, field
from typing import Optional, Union

import pandas as pd
from datasets import Dataset, DatasetDict

from dataquality.integrations.seq2seq.formatters.base import (
    BaseFormatter,
    DefaultFormatter,
)


@dataclass
class BaseAutoDatasetConfig:
    """Configuration for creating a dataset from a file or object

    One of `hf_name`, `train_path` or `train_dataset` should be provided. If none of
    those is, a demo dataset will be loaded by Galileo for training.

    :param hf_data: Union[DatasetDict, str] Use this param if you have huggingface
        data in the hub or in memory. Otherwise see `train_data`, `val_data`,
        and `test_data`. If provided, train_data, val_data, and test_data are ignored
    :param train_path: Optional path to training data file to use. Must be:
        * Path to a local file
    :param val_path: Optional path to validation data to use. Must be:
        * Path to a local file
    :param test_path: Optional test data to use. Must be:
        * Path to a local file
    :param train_data: Optional training data to use. Can be one of
        * Pandas dataframe
        * Huggingface dataset
    :param val_data: Optional validation data to use. Can be one of
        * Pandas dataframe
        * Huggingface dataset
    :param test_data: Optional test data to use. Can be one of
        * Pandas dataframe
        * Huggingface dataset
    :param input_col: Column name for input data, defaults to "text"
    :param target_col: Column name for target data, defaults to "label"
    """

    hf_data: Optional[Union[DatasetDict, str]] = None
    # If dataset provided as path to local file
    train_path: Optional[str] = None
    val_path: Optional[str] = None
    test_path: Optional[str] = None
    # If dataset provided as object
    train_data: Optional[Union[pd.DataFrame, Dataset]] = None
    val_data: Optional[Union[pd.DataFrame, Dataset]] = None
    test_data: Optional[Union[pd.DataFrame, Dataset]] = None
    # Column names
    input_col: str = "text"
    target_col: str = "label"
    # Dataset input / output formatter
    formatter: BaseFormatter = field(default_factory=DefaultFormatter)

    def __post_init__(self) -> None:
        if not any(
            [
                self.hf_data is not None,
                self.train_path is not None,
                self.train_data is not None,
            ]
        ):
            raise ValueError(
                "One of hf_data, train_path, or train_data must be provided."
                "To use a random demo dataset in `auto`, set dataset_config to None."
            )


@dataclass
class BaseAutoTrainingConfig:
    """Configuration for training a HuggingFace model

    Base config values are based on auto with Text Classification. Can be overridden
    by parent class for each modality.

    :param model: The pretrained AutoModel from huggingface that will be used to
        tokenize and train on the provided data. Default distilbert-base-uncased
    :param epochs: Optional num training epochs. If not set, we default to 15
    :param learning_rate: Optional learning rate. If not set, we default to 3e-4
    :param batch_size: Optional batch size. If not set, we default to 4
    :param create_data_embs: Whether to create data embeddings for this run. If set to
        None, data embeddings will be created only if a GPU is available
    :param return_model: Whether to return the trained model at the end of auto.
        Default False
    :param data_embs_col: Optional text col on which to compute data embeddings.
        If not set, we default to 'text'
    """

    model: str = "distilbert-base-uncased"
    epochs: int = 15
    learning_rate: float = 3e-4
    batch_size: int = 4
    create_data_embs: Optional[bool] = None
    data_embs_col: str = "text"
    return_model: bool = False
