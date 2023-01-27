import os
import sys
from typing import Dict, List, Optional

import numpy as np
import vaex
from vaex.dataframe import DataFrame

# from dataquality.loggers.base_logger import BaseGalileoLogger
from dataquality.clients.objectstore import ObjectStore
from dataquality.loggers.data_logger.base_data_logger import BaseGalileoDataLogger
from dataquality.loggers.logger_config.structured_classification import (
    structured_classification_logger_config,
)
from dataquality.schemas import __data_schema_version__
from dataquality.schemas.dataframe import BaseLoggerDataFrames
from dataquality.schemas.split import Split
from dataquality.utils import tqdm
from dataquality.utils.vaex import _save_hdf5_file

DATA_FOLDERS = ["prob", "data"]


class StructuredClassificationLogger(BaseGalileoDataLogger):
    __logger_name__ = "structured_classification"
    logger_config = structured_classification_logger_config

    def __init__(
        self,
    ) -> None:
        super().__init__()

    def validate(self) -> None:
        # Validate length of data, labels, and probs are the same
        assert len(self.X) == len(self.y) == len(self.probs)
        # Validate that the feature names are the same length as the number of features
        assert len(self.feature_names) == len(self.X[0])

    def log_samples(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        probs: np.ndarray,
        split: str = None,
        inference_name: str = None,
    ) -> None:
        self.X = X
        self.y = y
        self.feature_names = feature_names
        self.probs = probs
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
        /Users/username/.galileo/logs/proj-id/run-id/training/data.hdf5  # TODO: arrow?
        /Users/username/.galileo/logs/proj-id/run-id/training/probs.hdf5

        NOTE #2: We don't restrict row or feature counts here for cloud users. If we add
        that restriction, it will go here after gatting input_df
        """
        self.validate()
        # E.g. /Users/username/.galileo/logs/proj-id/run-id/training
        # E.g. /Users/username/.galileo/logs/proj-id/run-id/inference/my-inference
        write_dir = f"{self.write_output_dir}/{self.split}"
        if self.split == Split.inference:
            write_dir = f"{write_dir}/{self.inference_name}"

        os.makedirs(write_dir, exist_ok=True)

        data = self._get_data_dict()
        _save_hdf5_file(write_dir, "input_data.hdf5", data)

    def _get_data_dict(self) -> Dict:
        """Returns a dictionary with the input data

        Each of the features in the input data becomes a column in the DataFrame.
        We also add the gold label and other metadata to the Dict, such as
        the split, data schema version, and inference name.

        Example:
            >>> X = [[1, 2], [3, 4]]
            >>> y = [0, 1] # gold labels
            >>> features = ["feature_0", "feature_1"]
            >>> probs = [[0.9, 0.1], [0.1, 0.9]] # model predictions
            >>> split = "inference"
            >>> inference_name = "my-inference"
            >>> print(self._get_input_df())
            {
                "id": [0, 1],
                "feature_0": [1, 3],
                "feature_1": [2, 4],
                "gold": [0, 1],
                "probs": [[0.9, 0.1], [0.1, 0.9]],
                "split": ["inference", "inference"],
                "data_schema_version": [1, 1],
                "inference_name": ["my-inference", "my-inference"]
            }
        """
        num_samples = len(self.X)
        ids = list(range(num_samples))
        features = {
            feature: self.X[:, i] for i, feature in enumerate(self.feature_names)
        }
        data = {
            "id": ids,
            "gold": self.y,
            "prob": self.probs,
            "split": [self.split] * num_samples,
            "data_schema_version": [__data_schema_version__] * num_samples,
            **features,
        }
        if self.split == Split.inference:
            data.update(inference_name=self.inference_name)

        return data

    def upload(
        self, last_epoch: Optional[int] = None, create_data_embs: bool = False
    ) -> None:
        """
        Iterates through all of each splits [data/prob] and uploads
        them to Minio

        For structured data we don't use last_epoch or create_data_embs
        """
        self.check_for_logging_failures()
        print("☁️ Uploading Data")
        objectstore = ObjectStore()
        split_upload_folders = self.get_split_upload_folders()

        for split_path in tqdm(
            split_upload_folders,
            total=len(split_upload_folders),
            desc="Uploading splits",
            file=sys.stdout,
        ):
            df = vaex.open(f"{split_path}/input_data.hdf5")
            dfs = self.separate_dataframe(df)

            prob = dfs.prob
            data_df = dfs.data

            for data_folder, df_obj in tqdm(
                zip(DATA_FOLDERS, [data_df, prob]),
                total=len(DATA_FOLDERS),
                desc="Uploading Data",
                file=sys.stdout,
            ):
                ext = self.DATA_FOLDER_EXTENSION[data_folder]
                minio_file = f"{split_path}/{data_folder}/{data_folder}.{ext}"
                objectstore.create_project_run_object_from_df(
                    df=df_obj, object_name=minio_file
                )

    @classmethod
    def separate_dataframe(
        cls, df: DataFrame, prob_only: bool = True, split: str = None
    ) -> BaseLoggerDataFrames:
        """Separates the singular dataframe into prob and data components

        Gets the probability df, the embedding df, and the "data" df containing
        all other columns

        With example data:
            >>> X = [[1, 2], [3, 4]]
            >>> y = [1, 0] # gold labels
            >>> features = ["feature_0", "feature_1"]
            >>> probs = [[0.9, 0.1], [0.2, 0.8]] # model predictions
            >>> split = "training"

        Example prob df:
            >>> print(self.separate_df(df).prob)
            #    id    gold    prob
            0    0      1      [0.9, 0.1]
            1    1      0      [0.2, 0.8]

        Example data df:
            >>> print(self.separate_df(df).data)
            #    id    feature_0    feature_1    split    data_schema_version
            0    0            1            2    training    1
            1    1            3            4    training    1
        """
        df_copy = df.copy()
        prob_cols = ["id", "gold", "prob"]
        data_cols = [col for col in df_copy.get_column_names() if col not in prob_cols]
        data_cols += ["id"]

        data_df = df_copy[data_cols]
        prob_df = df_copy[prob_cols]

        # We don't use emb, but for linting it can't be None
        return BaseLoggerDataFrames(prob=prob_df, emb=df_copy, data=data_df)

    def get_split_upload_folders(self) -> List[str]:
        """Returns a list of all the split folders that need to be uploaded

        This returns a list containing paths to each split folder that was logged.
        For inference, we include a path to each inference folder that was logged.

        Example:
            >>> self.get_split_upload_folders()
            [
                "/Users/username/.galileo/logs/proj-id/run-id/training",
                "/Users/username/.galileo/logs/proj-id/run-id/validation",
                "/Users/username/.galileo/logs/proj-id/run-id/test",
                "/Users/username/.galileo/logs/proj-id/run-id/inference/my-inference",
                "/Users/username/.galileo/logs/proj-id/run-id/inference/my-other-inference",
            ]
        """
        upload_folders = []
        for split in Split.get_valid_attributes():
            split_dir = f"{self.write_output_dir}/{split}"
            split_logged = os.path.exists(split_dir)
            if not split_logged:
                continue

            if split == Split.inference:
                inf_names = os.listdir(split_dir)
                upload_folders.extend(
                    [f"{split_dir}/{inf_name}" for inf_name in inf_names]
                )
            else:
                upload_folders.append(split_dir)

        return upload_folders
