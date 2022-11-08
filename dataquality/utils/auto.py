import os
from random import choice
from typing import List, Optional, Union

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset

from dataquality.exceptions import GalileoException


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
                "Local file path extension must be readable by panda. "
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
