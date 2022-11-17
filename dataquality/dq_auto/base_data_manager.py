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
        """Makes sure at `train` or `training` are in dict, removes invalid keys"""
        valid_keys = Split.get_valid_keys()
        assert (
            "train" in dd or "training" in dd
        ), f"Must have `train` or `training` split in data, found {dd.keys()}"
        # Only keep valid split keys. Convert splits to enum Split
        dd_clean = DatasetDict({Split[k]: v for k, v in dd.items() if k in valid_keys})
        return dd_clean

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
            "Dataset must be one of pandas DataFrame, "
            "huggingface Dataset, or string path"
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
            # We don't need to check for train because `try_load_dataset_dict` validates
            # that it exists already. One of hf_data or train_data must exist
            dd[Split.train] = self._convert_to_hf_dataset(train_data, labels)
            if val_data is not None:
                dd[Split.validation] = self._convert_to_hf_dataset(val_data, labels)
            if test_data is not None:
                dd[Split.test] = self._convert_to_hf_dataset(test_data, labels)
        return self._validate_dataset_dict(dd, labels)
