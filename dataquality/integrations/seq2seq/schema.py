from dataclasses import dataclass
from typing import List, Optional, Union

import pandas as pd
from datasets import Dataset, DatasetDict


@dataclass
class AutoDatasetConfig:
    """Configuration for creating a dataset from a file or object

    One of `hf_name`, `train_path` or `train_dataset` should be provided. If none of
    those is, a demo dataset will be loaded by Galileo for training.


    :param hf_data: Union[DatasetDict, str] Use this param if you have huggingface
        data in the hub or in memory. Otherwise see `train_data`, `val_data`,
        and `test_data`. If provided, train_data, val_data, and test_data are ignored
    :param train_path: Optional path to training data file to use. Can be one of
        * Path to a local file
        * Huggingface dataset hub path
    :param val_path: Optional path to validation data to use. Can be one of
        * Path to a local file
        * Huggingface dataset hub path
    :param test_path: Optional test data to use. Can be one of
        * Path to a local file
        * Huggingface dataset hub path
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


@dataclass
class AutoTrainingConfig:
    """Configuration for training a seq2seq model

    :param model: The pretrained AutoModel from huggingface that will be used to
        tokenize and train on the provided data. Default google/flan-t5-base
    :param epochs: Optional num training epochs. If not set, we default to 3
    :param learning_rate: Optional learning rate. If not set, we default to 3e-4
    :param accumulation_steps: Optional accumulation steps. If not set, we default to 4
    :param batch_size: Optional batch size. If not set, we default to 4
    :param create_data_embs: Whether to create data embeddings for this run. Default
        False
    :param max_input_tokens: Optional max input tokens. If not set, we default to 512
    :param max_target_tokens: Optional max target tokens. If not set, we default to 128
    """

    model: str = "google/flan-t5-base"
    epochs: int = 3
    learning_rate: float = 3e-4
    accumulation_steps: int = 4
    batch_size: int = 4
    create_data_embs: Optional[bool] = None
    max_input_tokens: int = 512
    max_target_tokens: int = 128


@dataclass
class AutoGenerationConfig:
    """Configuration for generating insights from a trained seq2seq model

    We use the default values in HF GenerationConfig
    See more about the parameters here:
    https://huggingface.co/docs/transformers/v4.30.0/en/main_classes/text_generation#transformers.GenerationConfig

    :param generation_splits: Optional list of splits to generate on. If not set, we
        default to ["test"]
    """

    max_new_tokens: int = 16
    temperature: float = 0.2
    do_sample: bool = False  # Whether we use multinomial sampling
    top_p: float = 1.0
    top_k: int = 50
    generation_splits: Optional[List[str]] = None
