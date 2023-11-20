from dataclasses import dataclass
from typing import List, Optional

from dataquality.dq_auto.schema import BaseAutoDatasetConfig, BaseAutoTrainingConfig


@dataclass
class Seq2SeqDatasetConfig(BaseAutoDatasetConfig):
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
    :param input_col: Column name for input data, defaults to "input" for S2S
    :param target_col: Column name for target data, defaults to "target" for S2s
    """

    input_col: str = "input"
    target_col: str = "target"


@dataclass
class Seq2SeqTrainingConfig(BaseAutoTrainingConfig):
    """Configuration for training a seq2seq model

    :param model: The pretrained AutoModel from huggingface that will be used to
        tokenize and train on the provided data. Default google/flan-t5-base
    :param epochs: Optional num training epochs. If not set, we default to 3
    :param learning_rate: Optional learning rate. If not set, we default to 3e-4
    :param accumulation_steps: Optional accumulation steps. If not set, we default to 4
    :param batch_size: Optional batch size. If not set, we default to 4
    :param create_data_embs: Whether to create data embeddings for this run. If set to
        None, data embeddings will be created only if a GPU is available
    :param max_input_tokens: Optional max input tokens. If not set, we default to 512
    :param max_target_tokens: Optional max target tokens. If not set, we default to 128
    :param data_embs_col: Optional text col on which to compute data embeddings.
        If not set, we default to 'input', can also be set to `target` or
        `generated_output`
    """

    # Overwrite base values
    model: str = "google/flan-t5-base"
    epochs: int = 3
    # Custom Seq2Seq values
    accumulation_steps: int = 4
    max_input_tokens: int = 512
    max_target_tokens: int = 128
    # Data embeddings. Can also be set to `target` or `generated_output`
    data_embs_col: str = "input"


@dataclass
class Seq2SeqGenerationConfig:
    """Configuration for generating insights from a trained seq2seq model

    We use the default values in HF GenerationConfig
    See more about the parameters here:
    https://huggingface.co/docs/transformers/v4.30.0/en/main_classes/text_generation#transformers.GenerationConfig

    :param generation_splits: Optional list of splits to generate on. If not set, we
        default to ["test"]
    """

    max_new_tokens: int = 64
    temperature: float = 0.2
    do_sample: bool = False  # Whether we use multinomial sampling
    top_p: float = 1.0
    top_k: int = 50
    generation_splits: Optional[List[str]] = None
