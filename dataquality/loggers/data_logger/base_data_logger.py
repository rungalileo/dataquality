import os
import warnings
from abc import abstractmethod
from glob import glob
from typing import Any, Dict, List, Optional, Union
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
    _validate_unique_ids,
    concat_hdf5_files,
    drop_empty_columns,
    get_dup_ids,
    valid_ids,
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
            new_name = f"{write_input_dir}/{str(uuid4()).replace('-', '')[:12]}.arrow"
            os.rename(file_path, new_name)
            logged_data = vaex.open(new_name)
            combined_data = vaex.concat([logged_data, df])
            # Validate there are no duplicated IDs
            for split in df["split"].unique():
                split_df = df[df["split"] == split]
                if not valid_ids(split_df):
                    dups = get_dup_ids(split_df)
                    combined_data.close()
                    os.rename(new_name, file_path)  # Revert name, we aren't logging
                    raise GalileoException(
                        "It seems the newly logged data has IDs that duplicate "
                        f"previously logged data for split {split}. "
                        f"Duplicated IDs: {dups}"
                    )
            with vaex.progress.tree("vaex", title="Appending input data"):
                combined_data.export(file_path)
            os.remove(new_name)
        else:
            with vaex.progress.tree("vaex", title="Exporting input data"):
                df.export(file_path)
        df.close()

    @classmethod
    def upload(cls) -> None:
        """
        Iterates through all of the splits/epochs/[data/emb/prob] folders, concatenates
        all of the files with vaex, and uploads them to a single file in minio
        """
        ThreadPoolManager.wait_for_threads()
        print("☁️ Uploading Data")
        proj_run = f"{config.current_project_id}/{config.current_run_id}"
        location = f"{cls.LOG_FILE_DIR}/{proj_run}"
        object_store = ObjectStore()

        in_frame = vaex.open(
            f"{location}/{BaseGalileoDataLogger.INPUT_DATA_NAME}"
        ).copy()
        for split in Split.get_valid_attributes():
            split_loc = f"{location}/{split}"
            if not os.path.exists(split_loc):
                continue

            in_frame_split = in_frame[in_frame["split"].str.equals(split)].copy()
            # Drop any columns for this split that are empty
            # (ex: metadata logged for a different split)
            in_frame_split = drop_empty_columns(in_frame_split)
            # Remove the mask, work with only the filtered rows
            in_frame_split = in_frame_split.extract()
            for epoch_dir in glob(f"{split_loc}/*"):
                epoch = int(epoch_dir.split("/")[-1])
                # For all epochs that aren't the last 2 (early stopping), we only want
                # to upload the probabilities (for DEP calculation).
                if epoch < cls.logger_config.last_epoch - 1:
                    prob_only = True
                else:
                    prob_only = False

                str_cols = concat_hdf5_files(epoch_dir, prob_only)
                out_frame = vaex.open(f"{epoch_dir}/{HDF5_STORE}")
                # Post concat, string columns come back as bytes and need conversion
                for col in str_cols:
                    out_frame[col] = out_frame[col].to_arrow().cast(pa.large_string())
                if prob_only:
                    out_frame["epoch"] = vaex.vconstant(epoch, length=len(out_frame))
                    out_frame["split"] = vaex.vconstant(
                        split, length=len(out_frame), dtype="str"
                    )

                in_out_frames = cls.process_in_out_frames(
                    in_frame_split, out_frame, prob_only
                )
                prob = in_out_frames.prob
                emb = in_out_frames.emb
                data_df = in_out_frames.data

                for data_folder, df_obj in tqdm(
                    zip(DATA_FOLDERS, [emb, prob, data_df]), total=3, desc=split
                ):
                    if prob_only and data_folder != "prob":
                        continue
                    ext = cls.DATA_FOLDER_EXTENSION[data_folder]
                    minio_file = (
                        f"{proj_run}/{split}/{epoch}/{data_folder}/{data_folder}.{ext}"
                    )
                    object_store.create_project_run_object_from_df(
                        df=df_obj, object_name=minio_file
                    )

    @classmethod
    def process_in_out_frames(
        cls, in_frame: DataFrame, out_frame: DataFrame, prob_only: bool
    ) -> BaseLoggerInOutFrames:
        """Processes input and output dataframes from logging

        Validates uniqueness of IDs in the dataframes
        Joins inputs and outputs
        Splits the dataframes into prob, emb, and data for uploading to minio
        """
        _validate_unique_ids(out_frame)
        in_out = _join_in_out_frames(in_frame, out_frame)

        prob, emb, data_df = cls.split_dataframe(in_out, prob_only)
        return BaseLoggerInOutFrames(prob=prob, emb=emb, data=data_df)

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
    def split_dataframe(cls, df: DataFrame, prob_only: bool) -> DataFrame:
        ...

    @abstractmethod
    def _get_input_df(self) -> DataFrame:
        ...

    @classmethod
    def set_tagging_schema(cls, tagging_schema: TaggingSchema) -> None:
        """Sets the tagging schema, if applicable. Must be implemented by child"""
        raise GalileoException(f"Cannot set tagging schema for {cls.__logger_name__}")
