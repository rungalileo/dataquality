import os
import re
import warnings
import webbrowser
from datetime import datetime
from random import choice
from typing import List, Optional, Union

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from transformers import Trainer

import dataquality as dq
from dataquality.core.init import BAD_CHARS_REGEX
from dataquality.exceptions import GalileoException, GalileoWarning
from dataquality.integrations.transformers_trainer import watch
from dataquality.schemas.split import Split


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


def open_console_url(link: Optional[str] = "") -> None:
    """Tries to open the console url in the browser, if possible.

    This will work in local environments like jupyter or a python script, but won't
    work in colab (because colab is running on a server, so there's no "browser" to
    interact with). This also prints out the link for users to click so even in those
    environments they still have something to interact with.
    """
    if not link:
        return
    try:
        webbrowser.open(link)
    # In some environments, webbrowser will raise. Other times it fails silently (colab)
    except Exception:
        pass
    finally:
        print(f"Click here to see your run! {link}")


def do_train(trainer: Trainer, encoded_data: DatasetDict, wait: bool) -> None:
    watch(trainer)
    trainer.train()
    if Split.test in encoded_data:
        # We pass in a huggingface dataset but typing wise they expect a torch dataset
        trainer.predict(test_dataset=encoded_data[Split.test])  # type: ignore
    res = dq.finish(wait=wait) or {}
    open_console_url(res.get("link"))


def run_name_from_hf_dataset(name: str) -> str:
    name_today = f"{name}_{datetime.today()}"
    return re.sub(BAD_CHARS_REGEX, "_", name_today)
