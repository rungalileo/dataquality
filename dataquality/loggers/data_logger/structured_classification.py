from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import vaex
import xgboost as xgb
from vaex.dataframe import DataFrame

from dataquality.clients.objectstore import ObjectStore
from dataquality.loggers.data_logger.base_data_logger import BaseGalileoDataLogger
from dataquality.loggers.logger_config.structured_classification import (
    structured_classification_logger_config,
)
from dataquality.schemas import __data_schema_version__
from dataquality.schemas.split import Split

DATA_FOLDERS = ["prob", "data"]


class StructuredClassificationDataLogger(BaseGalileoDataLogger):
    __logger_name__ = "structured_classification"
    logger_config = structured_classification_logger_config

    def __init__(
        self,
        model: Optional[xgb.XGBClassifier] = None,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        probs: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__()
        self.model: xgb.XGBClassifier = (
            model if model is not None else xgb.XGBClassifier()
        )
        self.X: np.ndarray = X if X is not None else np.array([])
        self.y: np.ndarray = y if y is not None else np.array([])
        self.probs: np.ndarray = probs if probs is not None else np.array([])

    def validate(self) -> None:
        """Validates the input data before logging to Minio

        Validates:
            - The split is set and is one of the allowed splits (in super)
            - If the user is cloud the split is not inference (in super)
            -
            - The length of the data and probs are the same
        """
        super().validate()
        assert len(self.dataset) == len(
            self.probs
        ), "Data and probs are not the same length. "
        f"Data: {len(self.dataset)}, Probs: {len(self.probs)}"

    def create_dataset_from_samples(
        self, X: np.ndarray, y: Optional[np.ndarray], feature_names: List[str]
    ) -> pd.DataFrame:
        """Validates and creates pandas DataFrame from input data

        Validates:
            - The number of feature names is the same as the number of features
            - If the split is not inf, the length of the data and labels is the same

        Args:
            X: The data to be logged
            y: The labels to be logged
            feature_names: The names of the features in the data
        Returns:
            A pandas DataFrame with the data and labels
            The label column is named "gold"
        """
        feature_msg = (
            "The number of feature_names is not the same as the number of "
            "features in the datset."
        )
        assert len(feature_names) == len(X[0]), feature_msg

        dataset = pd.DataFrame(X, columns=feature_names)

        if y is not None and y.any():
            assert len(X) == len(y), "Data and labels are not the same length. "
            f"Data: {len(X)}, Labels: {len(y)}"
            dataset["gold"] = y

        return dataset

    def set_probs(self) -> None:
        """Sets the probs attribute for the class

        Assumes model and dataset are set.
        Assumes dataset is input data X, with optional "id" and "gold" columns
            that are removed
        """
        assert self.model is not None, "Model must be set before setting probs."
        assert self.dataset is not None, "Dataset must be set before setting probs."

        input_data = self.dataset.copy()
        # Drop gold and id columns if they exist
        input_data = input_data.drop(["gold", "id"], axis=1, errors="ignore").to_numpy()
        self.probs = self.model.predict_proba(input_data)

    def log_samples(
        self,
        model: xgb.XGBClassifier,
        X: np.ndarray,
        feature_names: List[str],
        y: Optional[np.ndarray],
        split: str = None,
        inference_name: str = None,
    ) -> None:
        self.model = model
        self.split = split
        self.inference_name = inference_name
        self.dataset = self.create_dataset_from_samples(X, y, feature_names)
        self.log()

    def log_structured_dataset(
        self,
        model: xgb.XGBClassifier,
        dataset: pd.DataFrame,
        label: Optional[str],
        split: str = None,
        inference_name: str = None,
    ) -> None:
        self.model = model
        self.dataset = dataset
        self.dataset.rename(columns={label: "gold"}, inplace=True)
        self.split = split
        self.inference_name = inference_name
        self.log()

    def log(self) -> None:
        """Writes input data to disk in .galileo/logs

        Note that unlike unstructured data, we don't batch data here, we write
        the entire input data to disk. If the input data is logged multiple times
        for the same split, we will overwrite the data on disk.

        Support for batching to come in V1 of structured data project.

        We write the input data to disk in the following locations:
        /Users/username/.galileo/logs/proj-id/run-id/training/data.hdf5
        /Users/username/.galileo/logs/proj-id/run-id/training/probs.hdf5

        NOTE #2: We don't restrict row or feature counts here for cloud users. If we add
        that restriction, it will go here after getting data_dict
        """
        self.set_probs()
        self.validate()
        # E.g. proj-id/run-id/training or proj-id/run-id/inference/my-inference
        object_base_path = f"{self.proj_run}/{self.split_name_path}"

        data_df, probs_df = self._get_dfs()

        print("☁️ Uploading Data")
        objectstore = ObjectStore()

        objectstore.create_project_run_object_from_df(
            data_df, f"{object_base_path}/data/data.hdf5"
        )
        objectstore.create_project_run_object_from_df(
            probs_df, f"{object_base_path}/prob/prob.hdf5"
        )

    def _get_dfs(self) -> Tuple[DataFrame, DataFrame]:
        """Returns data and probs as vaex DataFrames"""
        df = vaex.from_pandas(self.dataset)
        n_rows = len(df)
        ids = (
            df.id.to_numpy()
            if "id" in df.get_column_names()
            else self.dataset.index.to_numpy()
        )

        # Add id, split, data_schema_version, and inference_name to the data
        df["id"] = ids
        df["split"] = np.array([self.split] * n_rows)
        df["data_schema_version"] = np.array([__data_schema_version__] * n_rows)
        if self.split == Split.inference:
            df["inference_name"] = np.array([self.inference_name] * n_rows)

        # Create probs DataFrame
        probs_df = vaex.from_arrays(id=ids, prob=self.probs)
        if self.split != Split.inference and "gold" in self.dataset:
            probs_df["gold"] = self.dataset["gold"].to_numpy()

        return df, probs_df

    def upload(
        self, last_epoch: Optional[int] = None, create_data_embs: bool = False
    ) -> None:
        """Uploads the data and prob files for a given split to Minio

        For structured data we upload the data to Minio on log() instead of here.
        This is a noop for structured data.
        """
