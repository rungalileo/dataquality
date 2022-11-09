from typing import List, Optional, Union

import pandas as pd
from datasets import Dataset, DatasetDict

from dataquality.exceptions import GalileoException
from dataquality.schemas.split import Split
from dataquality.utils.auto import load_data_from_str, try_load_dataset_dict


class BaseDatasetManager:
    DEMO_DATASETS: List[str] = []

    def _validate_dataset_dict(
        self, dd: DatasetDict, labels: Optional[List[str]] = None
    ) -> DatasetDict:
        valid_keys = Split.get_valid_keys()
        for key in list(dd.keys()):
            assert key in valid_keys, (
                f"All keys of dataset must be one of {valid_keys}. "
                f"Found {list(dd.keys())}"
            )
        return dd

    def _convert_df_to_dataset(
        self, df: pd.DataFrame, labels: Optional[List[str]] = None
    ) -> Dataset:
        return Dataset.from_pandas(df)

    def _convert_to_hf_dataset(
        self,
        data: Union[pd.DataFrame, Dataset, str],
        labels: Optional[List[str]] = None,
    ) -> Dataset:
        """Loads the data into (hf) Dataset format.

        Data can be one of Dataset, pandas df, str. If str, it's either a path to a
        file or a path to a remote huggingface Dataset that we load with `load_dataset`
        """
        if isinstance(data, Dataset):
            return data
        if isinstance(data, pd.DataFrame):
            return self._convert_df_to_dataset(data, labels)
        if isinstance(data, str):
            ds = load_data_from_str(data)
            if isinstance(ds, pd.DataFrame):
                ds = self._convert_df_to_dataset(ds, labels)
            return ds
        raise GalileoException(
            "Dataset must be one of pandas df, huggingface Dataset, or string path"
        )

    def get_dataset_dict(
        self,
        hf_data: Union[DatasetDict, str] = None,
        train_data: Union[pd.DataFrame, Dataset, str] = None,
        val_data: Union[pd.DataFrame, Dataset, str] = None,
        test_data: Union[pd.DataFrame, Dataset, str] = None,
        labels: Optional[List[str]] = None,
    ) -> DatasetDict:
        """Creates and/or validates the DatasetDict provided by the user.

        If the user provides a DatasetDict, we simply validate it. Otherwise, we
        parse a combination of the parameters provided, generate a DatasetDict of their
        training data, and validate that.
        """
        dd = (
            try_load_dataset_dict(self.DEMO_DATASETS, hf_data, train_data)
            or DatasetDict()
        )
        if not dd:
            dd[Split.train] = self._convert_to_hf_dataset(train_data, labels)
            if val_data is not None:
                dd[Split.validation] = self._convert_to_hf_dataset(val_data, labels)
            if test_data is not None:
                dd[Split.test] = self._convert_to_hf_dataset(test_data, labels)
        return self._validate_dataset_dict(dd, labels)
