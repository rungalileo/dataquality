import os
from random import choice
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset

import dataquality as dq
from dataquality.auto.tc_trainer import get_trainer
from dataquality.exceptions import GalileoException
from dataquality.integrations.transformers_trainer import watch
from dataquality.schemas.split import Split
from dataquality.schemas.task_type import TaskType

DEMO_DATASETS = [
    "rungalileo/newsgroups",
    "rungalileo/trec6",
    "rungalileo/conv_intent",
    "rungalileo/emotion",
    "rungalileo/amazon_polarity_30k",
    "rungalileo/sst2",
]


def _get_dataset(data: Union[pd.DataFrame, Dataset, str]) -> Dataset:
    """Loads the data into (hf) Dataset format"""
    if isinstance(data, Dataset):
        return data
    if isinstance(data, pd.DataFrame):
        return Dataset.from_pandas(data)
    if isinstance(data, str):
        ext = os.path.splitext(data)[-1]
        if not ext:
            ds = load_dataset(data)
        else:
            func = f"from_{ext}"
            if not hasattr(Dataset, func):
                raise GalileoException(
                    "Local file path extension must be readable by huggingface Dataset."
                    f"Found {ext} which is not"
                )
            ds = getattr(Dataset, func)(data)
        assert isinstance(ds, Dataset), (
            f"Loaded data should be of type Dataset, but found {type(ds)}. If ds is a "
            f"DatasetDict, consider passing it to `hf_data`"
        )
        return ds
    raise GalileoException(
        "Dataset must be one of pandas df, huggingface Dataset, or string path"
    )


def _validate_dataset_dict(dd: DatasetDict) -> DatasetDict:
    valid_keys = ["train", "training", "test", "testing", "validation"]
    for key in list(dd.keys()):
        assert (
            key in valid_keys
        ), f"All keys of dataset must be one of {valid_keys}. Found {list(dd.keys())}"
        ds = dd[key]
        assert "text" in ds.features, "Dataset must have column `text`"
        assert "label" in ds.features, "Dataset must have column `label`"
        if "id" not in ds.features:
            dd[key] = ds.add_column("id", list(range(ds.num_rows)))
        # Use the split Enums
        dd[Split[key]] = dd.pop(key)
    return dd


def _get_dataset_dict(
    hf_data: Union[DatasetDict, str] = None,
    train_data: Union[pd.DataFrame, Dataset, str] = None,
    val_data: Union[pd.DataFrame, Dataset, str] = None,
    test_data: Union[pd.DataFrame, Dataset, str] = None,
) -> DatasetDict:
    dd = DatasetDict()
    if not any([hf_data, train_data]):
        hf_data = choice(DEMO_DATASETS)
        print(f"No dataset provided, using {hf_data} for run")
    if hf_data:
        ds = load_dataset(hf_data) if isinstance(hf_data, str) else hf_data
        assert isinstance(ds, DatasetDict), (
            "hf_data must be a path to a huggingface DatasetDict in the hf hub or a "
            "DatasetDict object. If this is just a Dataset, pass it to `train_data`"
        )
        dd = ds
    else:
        dd[Split.train] = _get_dataset(train_data)
        if val_data:
            dd[Split.validation] = _get_dataset(val_data)
        if test_data:
            dd[Split.test] = _get_dataset(test_data)
    return _validate_dataset_dict(dd)


def _get_labels(dd: DatasetDict, labels: Optional[List[str]] = None) -> List[str]:
    """Gets the labels for this dataset from the dataset if not provided.

    TODO: Is there any validation we need here?
    """
    if labels and isinstance(labels, (list, np.ndarray)):
        return list(labels)
    train_labels = dd[Split.train].features["label"]
    if hasattr(train_labels, "names"):
        return train_labels.names
    return sorted(set(dd[Split.train]["label"]))


def _log_dataset_dict(dd: DatasetDict) -> None:
    for key in dd:
        ds = dd[key]
        default_cols = ["text", "label", "id"]
        meta = [i for i in ds.features if i not in default_cols]
        dq.log_dataset(ds, meta=meta, split=key)


def auto(
    hf_data: Union[DatasetDict, str] = None,
    train_data: Union[pd.DataFrame, Dataset, str] = None,
    val_data: Union[pd.DataFrame, Dataset, str] = None,
    test_data: Union[pd.DataFrame, Dataset, str] = None,
    hf_model: str = "distilbert-base-uncased",
    labels: Optional[List[str]] = None,
    project_name: str = None,
    run_name: str = None,
    wait: bool = True,
) -> None:
    """Automatically gets insights on a text classification dataset

    Given either a pandas dataframe, file_path, or huggingface dataset location, this
    function will load the data, train a transformer, and get insights provided via a
    link to the Galileo console url

    One of `hf_data`, `train_data` should be provided. If neither of those are, a
    demo dataset will be loaded by Galileo for training.

    :param hf_data: Use this param if you have huggingface data in the hub
        or in memory. Otherwise see `train_data`, `val_data`, and `test_data`. If
        provided, train_data, val_data, and test_data will be ignored
    :param train_data: Optional training data to use. Can be one of
        * Pandas dataframe
        * Huggingface dataset
        * Path to a local file
        * Huggingface dataset hub name
    :param val_data: Optional validation data to use. Can be one of
        * Pandas dataframe
        * Huggingface dataset
        * Path to a local file
        * Huggingface dataset hub name
    :param test_data: Optional test data to use. Can be one of
        * Pandas dataframe
        * Huggingface dataset
        * Path to a local file
        * Huggingface dataset hub name
    :param hf_model: The automodel from huggingface. Default distilbert-base-uncased
    :param labels: Optional list of labels for this dataset. If not provided, they
        will be extracted from the data provided
    :param project_name: Optional project name. If not set, a random name will
        be generated
    :param run_name: Optional run name for this data. If not set, a random name will
        be generated
    :param wait: Whether to wait for Galileo to complete processing your run
    """
    dd = _get_dataset_dict(hf_data, train_data, val_data, test_data)
    labels = _get_labels(dd, labels)
    dq.login()
    dq.init(TaskType.text_classification, project_name=project_name, run_name=run_name)
    dq.set_labels_for_run(labels)
    _log_dataset_dict(dd)
    trainer, encoded_data = get_trainer(dd, labels, hf_model)
    watch(trainer)
    trainer.train()
    # TODO: What do we do with the test data? Do we call predict here?
    if Split.test in encoded_data:
        # We pass in a huggingface dataset but typing wise they expect a torch dataset
        trainer.predict(test_dataset=encoded_data[Split.test])  # type: ignore
    dq.finish(wait=wait)
