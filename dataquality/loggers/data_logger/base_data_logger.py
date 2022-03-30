import os
import warnings
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import uuid4

import numpy as np
import pyarrow as pa
import vaex
from vaex.dataframe import DataFrame

from dataquality.clients.objectstore import ObjectStore
from dataquality.core._config import config
from dataquality.exceptions import GalileoException
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


class BaseGalileoDataLogger(BaseGalileoLogger):
    MAX_META_COLS = 50  # Limit the number of metadata attrs a user can log
    MAX_STR_LEN = 50  # Max characters in a string metadata attribute
    INPUT_DATA_NAME = "input_data.arrow"

    DATA_FOLDER_EXTENSION = {data_folder: "hdf5" for data_folder in DATA_FOLDERS}

    def __init__(
        self, meta: Optional[Dict[str, List[Union[str, float, int]]]] = None
    ) -> None:
        super().__init__()
        self.meta: Dict[str, Any] = meta or {}

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
            with vaex.progress.tree("vaex", title="Exporting input data"):
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
            validate_ids_for_df(df)
        except GalileoException as e:  # Cleanup and raise on error
            merged_df.close()
            os.rename(tmp_name, file_path)  # Revert name, we aren't logging
            raise e

        with vaex.progress.tree("vaex", title="Appending input data"):
            merged_df.export(file_path)

        # Cleanup temporary file after appending input data
        os.remove(tmp_name)

    @classmethod
    def upload(cls) -> None:
        """
        Iterates through all of each splits children folders [data/emb/prob] for each
        inference name / epoch, concatenates all of the files with vaex, and uploads
        them to a single file in minio
        """
        ThreadPoolManager.wait_for_threads()
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

            in_frame_split = filter_df(in_frame, "split", split)
            cls.upload_split(object_store, in_frame_split, split, split_loc)

    @classmethod
    def upload_split(
        cls,
        object_store: ObjectStore,
        in_frame: DataFrame,
        split: str,
        split_loc: str,
    ) -> None:
        split_runs = os.listdir(split_loc)

        for split_run in split_runs:  # For each inference name or epoch
            in_frame_slice = in_frame.copy()
            prob_only = cls.prob_only(split, split_run)
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
    ) -> DataFrame:
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

        for data_folder, df_obj in tqdm(
            zip(DATA_FOLDERS, [emb, prob, data_df]), total=3, desc=split
        ):
            if df_obj.variables.get("skip_upload"):
                continue

            ext = cls.DATA_FOLDER_EXTENSION[data_folder]
            minio_file = (
                f"{proj_run}/{split}/{split_run}/" f"{data_folder}/{data_folder}.{ext}"
            )
            object_store.create_project_run_object_from_df(
                df=df_obj, object_name=minio_file
            )

    @classmethod
    def prob_only(cls, split: str, split_run: Union[int, str]) -> bool:
        if split == Split.inference:
            return False

        # If split is not inference, split_run must be epoch
        epoch = int(split_run)
        # For all epochs that aren't the last 2 (early stopping), we only
        # want to upload the probabilities (for DEP calculation).
        return bool(epoch < cls.logger_config.last_epoch - 1)

    @classmethod
    @abstractmethod
    def validate_labels(cls) -> None:
        ...

    def validate_metadata(self, batch_size: int) -> None:
        if len(self.meta.keys()) > self.MAX_META_COLS:
            warnings.warn(
                f"You can only log up to {self.MAX_META_COLS} metadata attrs. "
                f"The first {self.MAX_META_COLS} will be logged only."
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
                    f"will be removed."
                )
                continue
            bad_prefixes = ["galileo", "prob", "gold", "pred"]
            for bad_start in bad_prefixes:
                if key.startswith(bad_start):
                    warnings.warn(
                        "Metadata name must not start with the following "
                        f"prefixes: (galileo_, prob_, gold_. Won't log {key}"
                    )
                    continue
            # Must be the same length as input
            if len(values) != batch_size:
                warnings.warn(
                    f"Expected {batch_size} values for key {key} but got "
                    f"{len(values)}. Will not log this metadata column."
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
                    f"len < {self.MAX_STR_LEN} and numbers can be logged."
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

    @abstractmethod
    def _get_input_df(self) -> DataFrame:
        ...

    @classmethod
    def set_tagging_schema(cls, tagging_schema: TaggingSchema) -> None:
        """Sets the tagging schema, if applicable. Must be implemented by child"""
        raise GalileoException(f"Cannot set tagging schema for {cls.__logger_name__}")
