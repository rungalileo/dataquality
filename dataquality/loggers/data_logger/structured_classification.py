from __future__ import annotations

import os
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import vaex

from dataquality import config
from dataquality.clients.api import ApiClient
from dataquality.loggers.base_logger import BaseGalileoLogger
from dataquality.utils.name import validate_name

if TYPE_CHECKING:
    import xgboost as xgb

from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
from vaex.dataframe import DataFrame

from dataquality.clients.objectstore import ObjectStore
from dataquality.loggers.data_logger.base_data_logger import BaseGalileoDataLogger
from dataquality.loggers.logger_config.structured_classification import (
    StructuredClassificationLoggerConfig,
    structured_classification_logger_config,
)
from dataquality.schemas import __data_schema_version__
from dataquality.schemas.split import Split

api_client = ApiClient()


class StructuredClassificationDataLogger(BaseGalileoDataLogger):
    __logger_name__ = "structured_classification"
    logger_config: StructuredClassificationLoggerConfig = (
        structured_classification_logger_config
    )

    def __init__(
        self,
        model: Optional[xgb.XGBClassifier] = None,
        X: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y: Optional[Union[pd.Series, List, np.ndarray]] = None,
        feature_names: Optional[List[str]] = None,
        split: Optional[Split] = None,
        inference_name: Optional[str] = None,
    ) -> None:
        self.model = model
        self.X = X
        self.y = y
        self.feature_names = feature_names
        self.split = split
        self.inference_name = inference_name
        self.probs: Optional[np.ndarray] = None

    def validate_and_prepare_logger(self) -> None:
        """Validates the input data before logging to Minio

        Validates:
            - The model has a predict_proba method
            - The model is fit
            - The data is a pandas DataFrame or numpy array
            - If the split is not inf, the labels are a numpy array
            - If the split is not inf, the data and labels are the same length
            - If X is a numpy array, the feature names are provided
            - If X is a numpy array, the number of features in X and feature names
                are the same
            - Feature names match the feature names logged in a prior split
            - Feature names are valid names, no special chars

        Sets:
            - self.X to a pandas DataFrame if it is a numpy array
            - self.y to a numpy array if it is a list
            - self.feature_names to the column names of X if it is a pandas DataFrame
            - logger_config.feature_names to the column names of X if they aren't set
        """
        self.validate_and_format()

        assert self.model is not None, "Model must be included to log data."

        assert hasattr(self.model, "predict_proba"), (
            "Model must have a predict_proba method. "
            "If you are using a custom model, please implement a predict_proba method."
        )
        try:
            check_is_fitted(self.model)
        except NotFittedError:
            raise AssertionError("Model must be fit before logging data.")
        assert isinstance(
            self.X, (pd.DataFrame, np.ndarray)
        ), f"X must be a pandas DataFrame or numpy array, not {type(self.X)}"

        if self.split is not None and self.split != Split.inference:
            assert isinstance(self.y, (pd.Series, List, np.ndarray)), (
                "y must be a pandas Series, List, or numpy array of labels, "
                f"not {type(self.y)}"
            )
            self.y = np.array(self.y)
            assert len(self.X) == len(self.y), (
                "X and y must be the same length. "
                f"X has {len(self.X)} rows, y has {len(self.y)} rows"
            )

        if isinstance(self.X, np.ndarray):
            assert self.feature_names is not None, (
                "feature_names must be provided when logging X as a numpy array. "
                "If X is a pandas DataFrame, feature_names will be inferred from the "
                "column names."
            )
            assert len(self.X[0]) == len(
                self.feature_names
            ), "X and feature_names must have the same number of features"
            self.X = pd.DataFrame(self.X, columns=self.feature_names)
        elif isinstance(self.X, pd.DataFrame):
            self.feature_names = self.X.columns.tolist()

        if self.logger_config.feature_names:
            assert self.feature_names == self.logger_config.feature_names, (
                "feature_names must match the feature_names logged in a prior split."
                f"feature_names: {self.feature_names}"
                f"logger_config.feature_names: {self.logger_config.feature_names}"
            )
        else:
            assert self.feature_names is not None, (
                "feature_names must be provided when logging X as a numpy array. "
                "If X is a pandas DataFrame, feature_names will be inferred from the "
                "column names."
            )
            self.logger_config.feature_names = self.feature_names

        for feature in self.feature_names:
            validate_name(feature)

        if not self.logger_config.feature_importances and hasattr(
            self.model, "feature_importances_"
        ):
            self.logger_config.feature_importances = dict(
                zip(self.feature_names, self.model.feature_importances_.tolist())
            )

        self.set_probs()

    def set_probs(self) -> None:
        """Sets the probs attribute for the class

        Assumes model and dataset are set.
        """
        assert self.model is not None, "Model must be set before setting probs"
        assert self.X is not None, "X must be set before setting probs"
        self.probs = self.model.predict_proba(self.X)

    def save_feature_importances(self) -> None:
        """Saves feature importances in the DB

        Assumes the model is fit
        """
        template = " {inp} must be set before saving feature importances"
        assert config.current_project_id, print(template.format(inp="Project ID"))
        assert config.current_run_id, print(template.format(inp="Run ID"))

        api_client.set_metric_for_run(
            config.current_project_id,
            config.current_run_id,
            data={
                "key": "feature_importances",
                "value": 0,
                "epoch": 0,
                "extra": self.logger_config.feature_importances,
            },
        )

    def log(self) -> None:
        """Uploads data and probs df to disk in .galileo/logs

        Support for batching to come in V1 of structured data project.

        We write the dfs to disk in the following locations:
        /Users/username/.galileo/logs/proj-id/run-id/training/data/data.hdf5
        /Users/username/.galileo/logs/proj-id/run-id/training/prob/prob.hdf5

        NOTE: We don't restrict row or feature counts here for cloud users.
        """
        self.validate_and_prepare_logger()

        object_base_path = f"{self.write_output_dir}/{self.split_name_path}"
        df, prob_df = self._get_dfs()

        # Write DFs to disk
        os.makedirs(f"{object_base_path}/data", exist_ok=True)
        df.export(f"{object_base_path}/data/data.hdf5")
        os.makedirs(f"{object_base_path}/prob", exist_ok=True)
        prob_df.export(f"{object_base_path}/prob/prob.hdf5")

    def _get_dfs(self) -> Tuple[DataFrame, DataFrame]:
        """Returns data and prob as vaex DataFrames

        Data DF has:
        - id
        - pred
        - split
        - data_schema_version
        - inference_name (inference only)
        - **features

        Prob DF has:
        - id
        - prob
        - gold (non-inference only)
        """
        assert isinstance(self.X, pd.DataFrame), (
            "X must be a pandas DataFrame. " f"X is currently a {type(self.X)}"
        )
        assert self.probs is not None, "probs must be set before getting dfs"

        ids = self.X.id.values if "id" in self.X.columns else np.arange(len(self.X))
        n_rows = len(self.X)

        df = vaex.from_pandas(self.X)
        df["id"] = ids
        df["pred"] = np.argmax(self.probs, axis=1)
        df["split"] = np.array([self.split] * n_rows)
        df["data_schema_version"] = np.array([__data_schema_version__] * n_rows)
        if self.split == Split.inference:
            df["inference_name"] = np.array([self.inference_name] * n_rows)

        prob_df = vaex.from_arrays(
            id=ids,
            prob=self.probs,
        )
        if self.split != Split.inference:
            prob_df["gold"] = self.y

        return df, prob_df

    def upload(
        self, last_epoch: Optional[int] = None, create_data_embs: bool = False
    ) -> None:
        """Uploads the data and prob files for a given split to Minio

        Pulls files from disk and uploads them to Minio.

        From disk:
            /Users/username/.galileo/logs/proj-id/run-id/training/prob.hdf5
        To Minio:
            bucket/proj-id/run-id/training/prob.hdf5
        """
        self.save_feature_importances()

        print("☁️ Uploading Data")
        objectstore = ObjectStore()

        log_dir = self.write_output_dir
        # Traverse all files on disk for this proj/run
        for path, subdirs, files in os.walk(log_dir):
            for name in files:
                file_path = f"{path}/{name}"
                object_name = file_path.replace(
                    f"{BaseGalileoLogger.LOG_FILE_DIR}/", ""
                )
                objectstore.create_object(object_name=object_name, file_path=file_path)
