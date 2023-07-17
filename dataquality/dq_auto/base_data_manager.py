from typing import Dict, List, Optional, Union

import pandas as pd
from datasets import Dataset, DatasetDict

from dataquality.exceptions import GalileoException
from dataquality.schemas.split import Split
from dataquality.utils.auto import (
    _apply_column_mapping,
    load_data_from_str,
    try_load_dataset_dict,
)


class BaseDatasetManager:
    DEMO_DATASETS: List[str] = []

    def _validate_dataset_dict(
        self,
        dd: DatasetDict,
        inference_names: List[str],
        labels: Optional[List[str]] = None,
    ) -> DatasetDict:
        """Makes sure at `train` or `training` are in dict, removes invalid keys"""
        valid_splits = Split.get_valid_keys()
        valid_keys = valid_splits + inference_names
        assert (
            "train" in dd or "training" in dd
        ), f"Must have `train` or `training` split in data, found {dd.keys()}"
        # Only save valid keys + inference splits
        dd_pruned = DatasetDict({k: v for k, v in dd.items() if k in valid_keys})
        # Convert splits to enum Split if not inference
        dd_clean = DatasetDict(
            {Split[k] if k in valid_splits else k: v for k, v in dd_pruned.items()}
        )
        return dd_clean

    def _convert_df_to_dataset(
        self, df: pd.DataFrame, labels: Optional[List[str]] = None
    ) -> Dataset:
        return Dataset.from_pandas(df)

    def _convert_to_hf_dataset(
        self,
        data: Union[pd.DataFrame, Dataset, str],
        labels: Optional[List[str]] = None,
        column_mapping: Optional[Dict[str, str]] = None,
    ) -> Dataset:
        """Loads the data into (hf) Dataset format.

        Data can be one of Dataset, pandas df, str. If str, it's either a path to a
        file or a path to a remote huggingface Dataset that we load with `load_dataset`
        """
        ds = None
        if isinstance(data, Dataset):
            ds = data
        elif isinstance(data, pd.DataFrame):
            ds = self._convert_df_to_dataset(data, labels)
        elif isinstance(data, str):
            ds = load_data_from_str(data)
            if isinstance(ds, pd.DataFrame):
                ds = self._convert_df_to_dataset(ds, labels)
        if column_mapping is not None and ds is not None:
            ds = _apply_column_mapping(ds, column_mapping)

        if ds is None:
            raise GalileoException(
                "Dataset must be one of pandas DataFrame, "
                "huggingface Dataset, or string path"
            )
        return ds

    def get_dataset_dict(
        self,
        hf_data: Optional[Union[DatasetDict, str]] = None,
        hf_inference_names: Optional[List[str]] = None,
        train_data: Optional[Union[pd.DataFrame, Dataset, str]] = None,
        val_data: Optional[Union[pd.DataFrame, Dataset, str]] = None,
        test_data: Optional[Union[pd.DataFrame, Dataset, str]] = None,
        inference_data: Optional[Dict[str, Union[pd.DataFrame, Dataset, str]]] = None,
        labels: Optional[List[str]] = None,
        column_mapping: Optional[Dict[str, str]] = None,
    ) -> DatasetDict:
        """Creates and/or validates the DatasetDict provided by the user.

        If the user provides a DatasetDict, we simply validate it. Otherwise, we
        parse a combination of the parameters provided, generate a DatasetDict of their
        training data, and validate that.
        """
        hf_inference_names = hf_inference_names or []
        inf_names = []
        dd = (
            try_load_dataset_dict(self.DEMO_DATASETS, hf_data, train_data)
            or DatasetDict()
        )
        if dd:
            inf_names = [i for i in hf_inference_names if i in dd]
        else:
            # We don't need to check for train because `try_load_dataset_dict` validates
            # that it exists already. One of hf_data or train_data must exist
            dd[Split.train] = self._convert_to_hf_dataset(
                train_data, labels, column_mapping
            )
            if val_data is not None:
                dd[Split.validation] = self._convert_to_hf_dataset(
                    val_data, labels, column_mapping
                )
            if test_data is not None:
                dd[Split.test] = self._convert_to_hf_dataset(
                    test_data, labels, column_mapping
                )
            if inference_data is not None:
                for inf_name, inf_df in inference_data.items():
                    dd[inf_name] = self._convert_to_hf_dataset(
                        inf_df, labels, column_mapping
                    )
                    inf_names.append(inf_name)
        return self._validate_dataset_dict(dd, inf_names, labels)
