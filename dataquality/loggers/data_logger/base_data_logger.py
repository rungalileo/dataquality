import os
import warnings
from abc import abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Tuple, TypeVar, Union
from uuid import uuid4

import numpy as np
import pandas as pd
import pyarrow as pa
import vaex
from vaex.dataframe import DataFrame

from dataquality.clients.objectstore import ObjectStore
from dataquality.core._config import config
from dataquality.exceptions import GalileoException, GalileoWarning
from dataquality.loggers.base_logger import BaseGalileoLogger, BaseLoggerAttributes
from dataquality.schemas.dataframe import BaseLoggerInOutFrames
from dataquality.schemas.ner import TaggingSchema
from dataquality.schemas.split import Split
from dataquality.utils import tqdm
from dataquality.utils.hdf5_store import HDF5_STORE
from dataquality.utils.thread_pool import ThreadPoolManager
from dataquality.utils.vaex import (
    _join_in_out_frames,
    concat_hdf5_files,
    filter_df,
    validate_ids_for_df,
    validate_unique_ids,
)

DATA_FOLDERS = ["emb", "prob", "data"]
DataSet = TypeVar("DataSet", bound=Union[Iterable, pd.DataFrame, DataFrame])
MetasType = TypeVar("MetasType", bound=Dict[str, List[Union[str, float, int]]])
MetaType = TypeVar("MetaType", bound=Dict[str, Union[str, float, int]])
ITER_CHUNK_SIZE = 100_000


class BaseGalileoDataLogger(BaseGalileoLogger):
    MAX_META_COLS = 25  # Limit the number of metadata attrs a user can log
    MAX_STR_LEN = 100  # Max characters in a string metadata attribute
    INPUT_DATA_NAME = "input_data.arrow"

    DATA_FOLDER_EXTENSION = {data_folder: "hdf5" for data_folder in DATA_FOLDERS}

    def __init__(self, meta: MetasType = None) -> None:
        super().__init__()
        self.meta: Dict = meta or {}
        self.log_export_progress = True

    @abstractmethod
    def log_data_sample(self, *, text: str, id: int, **kwargs: Any) -> None:
        """Log a single input sample. See child for details"""

    @abstractmethod
    def log_data_samples(
        self, *, texts: List[str], ids: List[int], **kwargs: Any
    ) -> None:
        """Log a list of input samples. See child for details"""

    @abstractmethod
    def log_dataset(
        self,
        dataset: DataSet,
        *,
        batch_size: int = ITER_CHUNK_SIZE,
        text: Union[str, int] = "text",
        id: Union[str, int] = "id",
        split: Optional[Split] = None,
        meta: Optional[List[Union[str, int]]] = None,
        **kwargs: Any,
    ) -> None:
        """Log a dataset/iterable of input samples.

        Provide the dataset and the keys to index into it. See child for details"""

    def log(self) -> None:
        """Writes input data to disk in .galileo/logs

        If input data already exist, append new data to existing input file
        """
        self.validate()
        write_input_dir = (
            f"{BaseGalileoLogger.LOG_FILE_DIR}/{config.current_project_id}/"
            f"{config.current_run_id}"
        )
        if not os.path.exists(write_input_dir):
            os.makedirs(write_input_dir)

        df = self._get_input_df()
        file_path = f"{write_input_dir}/{BaseGalileoDataLogger.INPUT_DATA_NAME}"

        if os.path.isfile(file_path):
            self.append_input_data(df, write_input_dir, file_path)
        else:
            if self.log_export_progress:
                with vaex.progress.tree("vaex", title="Exporting input data"):
                    df.export(file_path)
            else:
                df.export(file_path)
        df.close()

    def append_input_data(
        self, df: DataFrame, write_input_dir: str, file_path: str
    ) -> None:
        # Create a temporary file for existing input data
        tmp_name = f"{write_input_dir}/{str(uuid4()).replace('-', '')[:12]}.arrow"
        os.rename(file_path, tmp_name)

        # Merge existing data with new data
        existing_df = vaex.open(tmp_name)
        merged_df = vaex.concat([existing_df, df])

        try:  # Validate there are no duplicated IDs
            validate_ids_for_df(merged_df)
        except GalileoException as e:  # Cleanup and raise on error
            merged_df.close()
            os.rename(tmp_name, file_path)  # Revert name, we aren't logging
            raise e

        if self.log_export_progress:
            with vaex.progress.tree("vaex", title="Appending input data"):
                merged_df.export(file_path)
        else:
            merged_df.export(file_path)

        # Cleanup temporary file after appending input data
        os.remove(tmp_name)

    @classmethod
    def upload(cls, last_epoch: Optional[int] = None) -> None:
        """
        Iterates through all of each splits children folders [data/emb/prob] for each
        inference name / epoch, concatenates all of the files with vaex, and uploads
        them to a single file in minio
        """
        ThreadPoolManager.wait_for_threads()
        cls.check_for_logging_failures()
        print("☁️ Uploading Data")
        object_store = ObjectStore()
        proj_run = f"{config.current_project_id}/{config.current_run_id}"
        location = f"{cls.LOG_FILE_DIR}/{proj_run}"

        in_frame = vaex.open(
            f"{location}/{BaseGalileoDataLogger.INPUT_DATA_NAME}"
        ).copy()
        for split in Split.get_valid_attributes():
            split_loc = f"{location}/{split}"
            if not os.path.exists(split_loc):
                continue
            if not len(in_frame[in_frame["split"] == split]):
                warnings.warn(
                    f"There was output data logged for split {split} but no input data "
                    "logged. Skipping upload for this split as there are no samples "
                    "to join to.",
                    GalileoWarning,
                )
                continue

            in_frame_split = filter_df(in_frame, "split", split)
            cls.upload_split(object_store, in_frame_split, split, split_loc, last_epoch)

    @classmethod
    def upload_split(
        cls,
        object_store: ObjectStore,
        in_frame: DataFrame,
        split: str,
        split_loc: str,
        last_epoch: Optional[int] = None,
    ) -> None:
        # If set, last_epoch will only let you upload to and including the provided
        # epoch value, nothing more.
        # If None, then slicing a list [:None] will include all values
        epochs_or_infs = os.listdir(split_loc)
        epochs_or_infs = sorted(
            epochs_or_infs, key=lambda i: int(i) if split != Split.inference else i
        )
        # last_epoch is inclusive
        last_epoch = last_epoch + 1 if last_epoch else last_epoch
        epochs_or_infs = epochs_or_infs[:last_epoch]

        # For each inference name or epoch of the given split
        for split_run in tqdm(epochs_or_infs, total=len(epochs_or_infs), desc=split):
            in_frame_slice = in_frame.copy()
            prob_only = cls.prob_only(epochs_or_infs, split, split_run)
            if split == Split.inference:
                in_frame_slice = filter_df(in_frame_slice, "inference_name", split_run)

            dir_name = f"{split_loc}/{split_run}"
            in_out_frames = cls.create_in_out_frames(
                in_frame_slice, dir_name, prob_only, split, split_run
            )
            cls.upload_in_out_frames(object_store, in_out_frames, split, split_run)

    @classmethod
    def create_in_out_frames(
        cls,
        in_frame: DataFrame,
        dir_name: str,
        prob_only: bool,
        split: str,
        split_run: Union[str, int],
    ) -> BaseLoggerInOutFrames:
        str_cols = concat_hdf5_files(dir_name, prob_only)
        out_frame = vaex.open(f"{dir_name}/{HDF5_STORE}")

        if split == Split.inference:
            dtype: Union[str, None] = "str"
            epoch_or_inf_name = "inference_name"
        else:
            dtype = None
            epoch_or_inf_name = "epoch"

        # Post concat, string columns come back as bytes and need conversion
        for col in str_cols:
            out_frame[col] = out_frame[col].to_arrow().cast(pa.large_string())
        if prob_only:
            out_frame["split"] = vaex.vconstant(
                split, length=len(out_frame), dtype="str"
            )
            out_frame[epoch_or_inf_name] = vaex.vconstant(
                split_run, length=len(out_frame), dtype=dtype
            )

        return cls.process_in_out_frames(
            in_frame, out_frame, prob_only, epoch_or_inf_name
        )

    @classmethod
    def process_in_out_frames(
        cls,
        in_frame: DataFrame,
        out_frame: DataFrame,
        prob_only: bool,
        epoch_or_inf_name: str,
    ) -> BaseLoggerInOutFrames:
        """Processes input and output dataframes from logging

        Validates uniqueness of IDs in the dataframes
        Joins inputs and outputs
        Splits the dataframes into prob, emb, and data for uploading to minio
        """
        validate_unique_ids(out_frame, epoch_or_inf_name)
        in_out = _join_in_out_frames(in_frame, out_frame)

        prob, emb, data_df = cls.split_dataframe(in_out, prob_only)
        # These df vars will be used in upload_in_out_frames
        emb.set_variable("skip_upload", prob_only)
        data_df.set_variable("skip_upload", prob_only)
        epoch_inf_val = out_frame[[epoch_or_inf_name]][0][0]
        prob.set_variable("progress_name", str(epoch_inf_val))

        return BaseLoggerInOutFrames(prob=prob, emb=emb, data=data_df)

    @classmethod
    def upload_in_out_frames(
        cls,
        object_store: ObjectStore,
        in_out_frames: BaseLoggerInOutFrames,
        split: str,
        split_run: Union[str, int],
    ) -> None:
        proj_run = f"{config.current_project_id}/{config.current_run_id}"

        prob = in_out_frames.prob
        emb = in_out_frames.emb
        data_df = in_out_frames.data

        epoch_inf = prob.variables.pop("progress_name", "")

        name = "inf_name" if split == Split.inference else "epoch"
        desc = f"{split} ({name}={epoch_inf})"
        for data_folder, df_obj in tqdm(
            zip(DATA_FOLDERS, [emb, prob, data_df]), total=3, desc=desc, leave=False
        ):
            if df_obj.variables.get("skip_upload"):
                continue

            ext = cls.DATA_FOLDER_EXTENSION[data_folder]
            minio_file = (
                f"{proj_run}/{split}/{split_run}/{data_folder}/{data_folder}.{ext}"
            )
            object_store.create_project_run_object_from_df(
                df=df_obj, object_name=minio_file
            )

    @classmethod
    def prob_only(
        cls, epochs: List[str], split: str, split_run: Union[int, str]
    ) -> bool:
        if split == Split.inference:
            return False

        # If split is not inference, split_run must be epoch
        epoch = int(split_run)
        # For all epochs that aren't the last 2 (early stopping), we only
        # want to upload the probabilities (for DEP calculation).
        max_epoch_for_split = max([int(i) for i in epochs])
        return bool(epoch < max_epoch_for_split - 1)

    def validate(self) -> None:
        self.set_split_epoch()

    @classmethod
    @abstractmethod
    def validate_labels(cls) -> None:
        ...

    def validate_metadata(self, batch_size: int) -> None:
        if len(self.meta.keys()) > self.MAX_META_COLS:
            warnings.warn(
                f"You can only log up to {self.MAX_META_COLS} metadata attrs. "
                f"The first {self.MAX_META_COLS} will be logged only.",
                GalileoWarning,
            )
        # When logging metadata columns, if the user breaks a rule, don't fail
        # completely, just warn them and remove that metadata column
        # Cast to list for in-place dictionary mutation
        reserved_keys = BaseLoggerAttributes.get_valid()
        valid_meta = {}
        for key, values in list(self.meta.items())[: self.MAX_META_COLS]:
            # Key must not override a default
            if key in reserved_keys:
                warnings.warn(
                    f"Metadata column names must not override default values "
                    f"{reserved_keys}. Metadata field {key} "
                    f"will be removed.",
                    GalileoWarning,
                )
                continue
            bad_prefixes = ["galileo", "prob", "gold", "pred"]
            for bad_start in bad_prefixes:
                if key.startswith(bad_start):
                    warnings.warn(
                        "Metadata name must not start with the following "
                        f"prefixes: (galileo_, prob_, gold_. Won't log {key}",
                        GalileoWarning,
                    )
                    continue
            # Must be the same length as input
            if len(values) != batch_size:
                warnings.warn(
                    f"Expected {batch_size} values for key {key} but got "
                    f"{len(values)}. Will not log this metadata column.",
                    GalileoWarning,
                )
                continue
            # Values must be a point, not an iterable
            valid_types = (str, int, float, np.floating, np.integer)
            invalid_values = filter(
                lambda t: not isinstance(t, valid_types)
                or (isinstance(t, str) and len(t) > self.MAX_STR_LEN),
                values,
            )
            bad_val = next(invalid_values, None)
            if bad_val:
                warnings.warn(
                    f"Metadata column {key} has one or more invalid values {bad_val} "
                    f"of type {type(bad_val)}. Only strings of "
                    f"len < {self.MAX_STR_LEN} and numbers can be logged.",
                    GalileoWarning,
                )
                continue
            valid_meta[key] = values
        self.meta = valid_meta

    @staticmethod
    def get_data_logger_attr(cls: object) -> str:
        """
        Returns the attribute that corresponds to the logger in the class.
        This assumes only 1 logger object exists in the class

        :param cls: The class
        :return: The attribute name
        """
        for attr in dir(cls):
            member_class = getattr(cls, attr)
            if isinstance(member_class, BaseGalileoDataLogger):
                return attr
        raise AttributeError("No data logger attribute found!")

    @classmethod
    @abstractmethod
    def split_dataframe(
        cls, df: DataFrame, prob_only: bool
    ) -> Tuple[DataFrame, DataFrame, DataFrame]:
        ...

    def validate_kwargs(self, kwargs: Dict) -> None:
        """Raises if a function that shouldn't get kwargs gets any"""
        if kwargs.keys():
            raise GalileoException(f"Unexpected arguments: {tuple(kwargs.keys())}")

    @abstractmethod
    def _get_input_df(self) -> DataFrame:
        ...

    @classmethod
    def set_tagging_schema(cls, tagging_schema: TaggingSchema) -> None:
        """Sets the tagging schema, if applicable. Must be implemented by child"""
        raise GalileoException(f"Cannot set tagging schema for {cls.__logger_name__}")
