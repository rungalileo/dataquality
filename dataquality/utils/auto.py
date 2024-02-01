import logging
import os
import re
import warnings
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Set, Union

import pandas as pd
from datasets import ClassLabel, Dataset, DatasetDict, load_dataset

from dataquality.dq_auto.schema import BaseAutoDatasetConfig
from dataquality.exceptions import GalileoException, GalileoWarning
from dataquality.schemas.split import Split
from dataquality.schemas.task_type import TaskType
from dataquality.utils.name import BAD_CHARS_REGEX


def sample_dataset_dict(
    dd: DatasetDict,
    dataset_config: BaseAutoDatasetConfig,
    max_train_size: Optional[int] = None,
) -> DatasetDict:
    """Samples the dataset dict to the max train size

    A few important notes:
    - If max train size is greater than the dataset size, we don't sample
    - If max train size is None we also don't sample
    - We set max eval size to be 25% of max train size
    - Test and inference data are not sampled
    """
    max_train_sz = max_train_size or dataset_config.formatter.max_train_size
    if not max_train_sz:
        return dd

    max_eval_sz = int(max_train_sz * 0.25)
    for split, dataset in dd.items():
        sampled_size = len(dataset)
        if Split[split] == Split.training:
            sampled_size = min(sampled_size, max_train_sz)
        elif Split[split] == Split.validation:
            sampled_size = min(sampled_size, max_eval_sz)

        if len(dataset) > sampled_size:
            # Slice the dataset to the max size
            dataset = dataset.select(range(sampled_size))
            dd[split] = dataset

    return dd


def get_meta_cols(
    cols: Iterable, reserved_cols: Optional[Set[str]] = None
) -> List[str]:
    """Returns the meta columns of a dataset."""
    reserved_cols = reserved_cols or set()
    default_cols = {"text", "label", "id"}
    default_cols = set(reserved_cols).union(default_cols)
    meta_columns = [col for col in cols if col not in default_cols]
    return list(meta_columns)


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
    elif ext == ".jsonl":
        # If it's a jsonl file, we load it as a pandas dataframe
        return pd.read_json(data, lines=True)
    else:
        # .csv -> read_csv, .parquet -> read_parquet
        func = f"read_{ext.lstrip('.')}"
        if not hasattr(pd, func):
            raise GalileoException(
                "Local file path extension must be readable by pandas. "
                f"Found {ext} which is not"
            )
        return getattr(pd, func)(data)


def add_val_data_if_missing(
    dd: DatasetDict, task_type: Optional[TaskType] = None
) -> DatasetDict:
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
    label_col: Optional[str]
    for col in ["tags", "ner_tags", "label"]:
        if col in ds_train.features:
            label_col = col
            break
    is_classlabel = False
    if task_type and task_type != TaskType.seq2seq:
        assert label_col in ds_train.features, "Must have label, ner_tags, or tags"
        # Can only stratify by a ClassLabel
        is_classlabel = isinstance(ds_train.features[label_col], ClassLabel)
    ds_train_test = ds_train.train_test_split(
        train_size=0.8, seed=42, stratify_by_column=label_col if is_classlabel else None
    )
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
    hf_data: Optional[Union[DatasetDict, str]] = None,
    train_data: Optional[Union[pd.DataFrame, Dataset, str]] = None,
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


def set_global_logging_level(
    level: int = logging.ERROR, prefices: Optional[List[str]] = None
) -> None:
    """
    Override logging levels of different modules based on their name as a prefix.
    It needs to be invoked after the modules have been loaded so that their loggers
    have been initialized.

    Src: https://github.com/huggingface/transformers/issues/3050#issuecomment-682167272

    Args:
        - level: desired level. e.g. logging.INFO. Optional. Default is logging.ERROR
        - prefices: list of one or more str prefices to match
          (e.g. ["transformers", "torch"]). Optional.
          Default is `[""]` to match all active loggers.
          The match is a case-sensitive `module_name.startswith(prefix)`
    """
    prefices = prefices or [""]
    prefix_re = re.compile(rf'^(?:{ "|".join(prefices) })')
    for name in logging.root.manager.loggerDict:
        if re.match(prefix_re, name):
            logging.getLogger(name).setLevel(level)


def add_class_label_to_dataset(
    ds: Dataset, labels: Optional[List[str]] = None
) -> Dataset:
    """Map a not ClassLabel 'label' column to a ClassLabel, if possible"""
    if "label" not in ds.features or isinstance(ds.features["label"], ClassLabel):
        return ds
    labels = labels if labels is not None else sorted(set(ds["label"]))
    # For string columns, map the label2idx so we can cast to ClassLabel
    if ds.features["label"].dtype == "string":
        label_to_idx = dict(zip(labels, range(len(labels))))
        ds = ds.map(lambda row: {"label": label_to_idx[row["label"]]})

    # https://github.com/python/mypy/issues/6239
    class_label = ClassLabel(num_classes=len(labels), names=labels)  # type: ignore
    ds = ds.cast_column("label", class_label)
    return ds


def _apply_column_mapping(dataset: Dataset, column_mapping: Dict[str, str]) -> Dataset:
    """
    Applies the provided column mapping to the dataset, renaming columns accordingly.
    """
    if isinstance(dataset, dict):
        dataset = Dataset.from_dict(dataset)
    # Making sure the keys of the column mapping are in the dataset
    # otherwise show a warning message and remove the key from the mapping
    clean_column_mapping = {**column_mapping}
    for key in list(clean_column_mapping.keys()):
        if key not in dataset.column_names:
            print(
                f"Column '{key}' in the column_mapping "
                "was not found in the dataset, so it was ignored."
            )
            clean_column_mapping.pop(key)

    dataset = dataset.rename_columns(clean_column_mapping)
    dset_format = dataset.format
    dataset = dataset.with_format(
        type=dset_format["type"],
        columns=dataset.column_names,
        output_all_columns=dset_format["output_all_columns"],
        **dset_format["format_kwargs"],
    )
    return dataset
