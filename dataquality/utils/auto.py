import os
import re
import warnings
from datetime import datetime
from random import choice
from typing import List, Optional, Union

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset

from dataquality.core.init import BAD_CHARS_REGEX
from dataquality.exceptions import GalileoException, GalileoWarning
from dataquality.schemas.split import Split
from dataquality.schemas.task_type import TaskType


def load_data_from_str(data: str) -> Union[pd.DataFrame, Dataset]:
    """Loads string data from either hf or disk.

    The string corresponds to either a path to a local file or a path to a remote
    huggingface Dataset that we load with `load_dataset`

    We can tell what it is based on if there's an extension at the end of the file
    """
    ext = os.path.splitext(data)[-1]
    if not ext:
        # If there is no file extension, it's a huggingface Dataset, so we load it from
        # huggingface hub
        ds = load_dataset(data)
        assert isinstance(ds, Dataset), (
            f"Loaded data should be of type Dataset, but found {type(ds)}. If ds is a "
            f"DatasetDict, consider passing it to `hf_data` (dq.auto(hf_data=data))"
        )
        return ds
    else:
        # .csv -> read_csv, .parquet -> read_parquet
        func = f"read_{ext.lstrip('.')}"
        if not hasattr(pd, func):
            raise GalileoException(
                "Local file path extension must be readable by pandas. "
                f"Found {ext} which is not"
            )
        return getattr(pd, func)(data)


def try_load_dataset_dict(
    demo_datasets: List[str],
    hf_data: Union[DatasetDict, str] = None,
    train_data: Union[pd.DataFrame, Dataset, str] = None,
) -> Optional[DatasetDict]:
    """Tries to load the DatasetDict if available

    If the user provided the hf_data param we load it from huggingface
    If they provided nothing, we load the demo dataset
    Otherwise, we return None, because the user provided train/test/val data, and that
    requires task specific processing
    """
    if all([hf_data is None, train_data is None]):
        hf_data = choice(demo_datasets)
        print(f"No dataset provided, using {hf_data} for run")
    if hf_data:
        ds = load_dataset(hf_data) if isinstance(hf_data, str) else hf_data
        assert isinstance(ds, DatasetDict), (
            "hf_data must be a path to a huggingface DatasetDict in the hf hub or a "
            "DatasetDict object. If this is just a Dataset, pass it to `train_data`"
        )
        return ds
    return None


def add_val_data_if_missing(dd: DatasetDict) -> DatasetDict:
    """Splits user provided training data if missing

    We need validation data in order to train a model properly, and it's required to
    enable early stopping. If the user didn't provide that val data, simply split
    their train data 80/20 for validation data.

    If there is test data, we can use the test as our val data (because that's also
    pretty common)
    """
    if Split.validation in dd:
        return dd
    if Split.test in dd:
        dd[Split.validation] = dd.pop(Split.test)
        return dd
    warnings.warn(
        "No validation data was provided. Train data will be split into train/val "
        "automatically. To avoid this in the future, add validation data `val_data` to "
        "your dataset. ",
        GalileoWarning,
    )
    ds_train = dd[Split.train]
    ds_train_test = ds_train.train_test_split(train_size=0.8, seed=42)
    dd[Split.train] = ds_train_test["train"]
    dd[Split.validation] = ds_train_test["test"]
    return dd


def run_name_from_hf_dataset(name: str) -> str:
    name_today = f"{name}_{datetime.today()}"
    return re.sub(BAD_CHARS_REGEX, "_", name_today)


def _get_task_type_from_cols(cols: List[str]) -> TaskType:
    if "text" in cols and "label" in cols:
        return TaskType.text_classification
    elif "tokens" in cols and ("tags" in cols or "ner_tags" in cols):
        return TaskType.text_ner
    else:
        raise GalileoException(
            "Data must either have `text` and `label` for text classification or "
            f"`tokens` and `tags` (or `ner_tags`) for NER. Yours had {cols}"
        )


def _get_task_type_from_hf(data: Union[DatasetDict, str]) -> TaskType:
    """Gets the task type from the huggingface data

    We get down to a Dataset object so we can inspect the columns
    """
    # If it's a string, download from huggingface
    hf_data = load_dataset(data) if isinstance(data, str) else data
    # DatasetDict is just a child of dict
    assert isinstance(hf_data, dict), (
        "hf_data should be a DatasetDict (or path to one). If this is a Dataset, pass "
        "it to train_data"
    )
    ds = hf_data[next(iter(hf_data))]
    return _get_task_type_from_cols(list(ds.features))


def _get_task_type_from_train(
    train_data: Union[pd.DataFrame, Dataset, str]
) -> TaskType:
    data = load_data_from_str(train_data) if isinstance(train_data, str) else train_data
    if isinstance(data, Dataset):
        return _get_task_type_from_cols(list(data.features))
    else:
        return _get_task_type_from_cols(list(data.columns))


def get_task_type_from_data(
    hf_data: Union[DatasetDict, str] = None,
    train_data: Union[pd.DataFrame, Dataset, str] = None,
) -> TaskType:
    """Determines the task type of the dataset by the dataset contents

    Text classification will have `text` and `label` and NER will have `tokens` and
    `tags`/`ner_tags`

    We know that one of these two parameters will be not None because that is validated
    before calling this function. See `dq.auto`
    """
    if hf_data is not None:
        return _get_task_type_from_hf(hf_data)
    return _get_task_type_from_train(train_data)
