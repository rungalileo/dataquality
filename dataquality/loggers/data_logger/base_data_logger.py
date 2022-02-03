import os
import warnings
from abc import abstractmethod
from glob import glob
from typing import Any, Dict, List, Optional, Union

import numpy as np
import vaex
from vaex.dataframe import DataFrame

from dataquality.clients.objectstore import ObjectStore
from dataquality.core._config import config
from dataquality.loggers.base_logger import BaseGalileoLogger, BaseLoggerAttributes
from dataquality.schemas.split import Split
from dataquality.utils import tqdm
from dataquality.utils.thread_pool import ThreadPoolManager
from dataquality.utils.vaex import _join_in_out_frames, _validate_unique_ids

DATA_FOLDERS = ["emb", "prob", "data"]


class BaseGalileoDataLogger(BaseGalileoLogger):
    MAX_META_COLS = 50  # Limit the number of metadata attrs a user can log
    MAX_STR_LEN = 50  # Max characters in a string metadata attribute
    INPUT_DATA_NAME = "input_data.arrow"

    def __init__(
        self, meta: Optional[Dict[str, List[Union[str, float, int]]]] = None
    ) -> None:
        super().__init__()
        self.meta: Dict[str, Any] = meta or {}

    @abstractmethod
    def validate(self) -> None:
        ...

    @abstractmethod
    def log(self) -> None:
        ...

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

            for epoch_dir in glob(f"{split_loc}/*"):
                epoch = int(epoch_dir.split("/")[-1])

                out_frame = vaex.open(f"{epoch_dir}/*")
                _validate_unique_ids(out_frame)
                in_out = _join_in_out_frames(in_frame, out_frame)

                prob, emb, data_df = cls.split_dataframe(in_out)

                for data_folder, df_obj in tqdm(
                    zip(DATA_FOLDERS, [emb, prob, data_df]), total=3, desc=split
                ):
                    minio_file = (
                        f"{proj_run}/{split}/{epoch}/{data_folder}/{data_folder}.hdf5"
                    )
                    object_store.create_project_run_object_from_df(
                        df=df_obj, object_name=minio_file
                    )

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
        Returns the attribute that corresponds to the GalileoDataConfig class.
        This assumes only 1 GalileoDataConfig object exists in the class

        :param cls: The class
        :return: The attribute name
        """
        for attr in dir(cls):
            member_class = getattr(cls, attr)
            if isinstance(member_class, BaseGalileoDataLogger):
                return attr
        raise AttributeError("No GalileoDataConfig attribute found!")

    @classmethod
    @abstractmethod
    def split_dataframe(cls, df: DataFrame) -> DataFrame:
        ...
