import os
from random import choice
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype
from datasets import Dataset, DatasetDict, load_dataset, ClassLabel

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


def _process_pandas_df(df: pd.DataFrame, labels: List[str] = None) -> Dataset:
    label_is_str = is_string_dtype(df["label"]) if "label" in df else False
    if not label_is_str:
        return Dataset.from_pandas(df)
    labels = labels if labels is not None else sorted(set(df["label"].tolist()))
    label_to_idx = dict(zip(labels, list(range(len(labels)))))
    df["label"] = df["label"].apply(lambda label: label_to_idx[label])
    class_label = ClassLabel(num_classes=len(labels), names=labels)
    ds = Dataset.from_pandas(df)
    ds = ds.cast_column("label", class_label)
    return ds

def _add_class_label_to_dataset(ds: Dataset, labels: List[str] = None) -> Dataset:
    """Map a not ClassLabel 'label' column to a ClassLabel"""
    if ds.features["label"].dtype == "string":
        return ds.class_encode_column("label", include_nulls=True)
    labels = labels if labels is not None else sorted(set(ds["label"]))
    class_label = ClassLabel(num_classes=len(labels), names=labels)
    ds = ds.cast_column("label", class_label)
    return ds


def _get_dataset(data: Union[pd.DataFrame, Dataset, str], labels: List[str] = None) -> Dataset:
    """Loads the data into (hf) Dataset format"""
    if isinstance(data, Dataset):
        return data
    if isinstance(data, pd.DataFrame):
        return _process_pandas_df(data, labels)
    if isinstance(data, str):
        ext = os.path.splitext(data)[-1]
        if not ext:
            ds = load_dataset(data)
        else:
            func = f"read_{ext}"
            if not hasattr(Dataset, func):
                raise GalileoException(
                    "Local file path extension must be readable by pandas."
                    f"Found {ext} which is not"
                )
            df = getattr(pd, func)(data)
            ds = _process_pandas_df(df, labels)
        assert isinstance(ds, Dataset), (
            f"Loaded data should be of type Dataset, but found {type(ds)}. If ds is a "
            f"DatasetDict, consider passing it to `hf_data` (dq.auto(hf_data=data))"
        )
        return ds
    raise GalileoException(
        "Dataset must be one of pandas df, huggingface Dataset, or string path"
    )


def _validate_dataset_dict(dd: DatasetDict, labels: List[str] = None) -> DatasetDict:
    valid_keys = ["train", "training", "test", "testing", "validation"]
    for key in list(dd.keys()):
        assert (
            key in valid_keys
        ), f"All keys of dataset must be one of {valid_keys}. Found {list(dd.keys())}"
        ds = dd.pop(key)
        assert "text" in ds.features, "Dataset must have column `text`"
        assert "label" in ds.features, "Dataset must have column `label`"
        if "id" not in ds.features:
            ds = ds.add_column("id", list(range(ds.num_rows)))
        if not isinstance(ds.features["label"], ClassLabel):
            ds = _add_class_label_to_dataset(ds, labels)
        # Use the split Enums
        dd[Split[key]] = ds
    return dd


def _get_dataset_dict(
    hf_data: Union[DatasetDict, str] = None,
    train_data: Union[pd.DataFrame, Dataset, str] = None,
    val_data: Union[pd.DataFrame, Dataset, str] = None,
    test_data: Union[pd.DataFrame, Dataset, str] = None,
    labels: List[str] = None
) -> DatasetDict:
    dd = DatasetDict()
    if not any([hf_data is not None, train_data is not None]):
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
        dd[Split.train] = _get_dataset(train_data, labels)
        if val_data is not None:
            dd[Split.validation] = _get_dataset(val_data, labels)
        if test_data is not None:
            dd[Split.test] = _get_dataset(test_data, labels)
    return _validate_dataset_dict(dd, labels)


def _get_labels(dd: DatasetDict, labels: Optional[List[str]] = None) -> List[str]:
    """Gets the labels for this dataset from the dataset if not provided.

    TODO: Is there any validation we need here?
    """
    if labels is not None and isinstance(labels, (list, np.ndarray)):
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
    labels: List[str] = None,
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
    dd = _get_dataset_dict(hf_data, train_data, val_data, test_data, labels)
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
