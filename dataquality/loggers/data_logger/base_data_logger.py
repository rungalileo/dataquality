import gc
import glob
import os
import sys
import warnings
from abc import abstractmethod
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, TypeVar, Union

import numpy as np
import pandas as pd
import vaex
from huggingface_hub.utils import HfHubHTTPError
from vaex.dataframe import DataFrame

from dataquality.clients.objectstore import ObjectStore
from dataquality.core._config import config
from dataquality.exceptions import GalileoException, GalileoWarning
from dataquality.loggers.base_logger import BaseGalileoLogger, BaseLoggerAttributes
from dataquality.schemas.dataframe import BaseLoggerDataFrames, DFVar
from dataquality.schemas.ner import TaggingSchema
from dataquality.schemas.split import Split
from dataquality.utils import tqdm
from dataquality.utils.cloud import is_galileo_cloud
from dataquality.utils.cuda import cuml_available
from dataquality.utils.emb import (
    DATA_EMB_PATH,
    apply_umap_to_embs,
    upload_umap_data_embs,
)
from dataquality.utils.file import _shutil_rmtree_retry
from dataquality.utils.helpers import galileo_verbose_logging
from dataquality.utils.thread_pool import ThreadPoolManager
from dataquality.utils.vaex import (
    _join_in_out_frames,
    create_data_embs_df,
    filter_df,
    get_output_df,
    validate_unique_ids,
)

DATA_FOLDERS = ["emb", "prob", "data"]
DataSet = TypeVar("DataSet", bound=Union[Iterable, pd.DataFrame, DataFrame])
MetasType = TypeVar("MetasType", bound=Dict[str, List[Union[str, float, int]]])
MetaType = TypeVar("MetaType", bound=Dict[str, Union[str, float, int]])
ITER_CHUNK_SIZE = 100_000


# Vaex issue https://github.com/vaexio/vaex/issues/2282
try:
    vaex.progress.bar("vaex")
except NameError:
    vaex.progress._progressbar_registry.registry["vaex"] = vaex.progress.simple


class BaseGalileoDataLogger(BaseGalileoLogger):
    """Base class for data loggers.

    A document col is a large str > 1k chars < 10k chars
    To avoid massive files, we limit the number of documents logged
    """

    MAX_META_COLS = 25  # Limit the number of metadata attrs a user can log
    MAX_STR_LEN = 1_000  # Max characters in a string metadata attribute
    MAX_DOC_LEN = 10_000  # Max characters in document metadata attribute
    LIMIT_NUM_DOCS = 3  # Limit the number of documents logged per split
    INPUT_DATA_BASE = "input_data"
    MAX_DATA_SIZE_CLOUD = 300_000
    # 2GB max size for arrow strings. We use 1.5GB for some buffer
    # https://issues.apache.org/jira/browse/ARROW-17828
    STRING_MAX_SIZE_B = 1.5e9

    DATA_FOLDER_EXTENSION = {data_folder: "hdf5" for data_folder in DATA_FOLDERS}
    INPUT_DATA_FILE_EXT = "arrow"

    def __init__(self, meta: Optional[MetasType] = None) -> None:
        super().__init__()
        self.meta: Dict = meta or {}
        self.log_export_progress = True

    @property
    def input_data_path(self) -> str:
        """Return the path to the input data folder.

        Example:
            /Users/username/.galileo/logs/proj-id/run-id/input_data
        """
        return f"{self.write_output_dir}/{BaseGalileoDataLogger.INPUT_DATA_BASE}"

    def input_data_file(
        self, input_num: Optional[int] = None, split: Optional[str] = None
    ) -> str:
        """Return the path to the input data file.

        Example:
            /Users/username/.galileo/logs/proj-id/run-id/input_data/train/data_0.arrow
        """
        if not split:
            assert self.split
            split = str(self.split)
        if input_num is None:
            # input_data_logged is a dict of {split: input_num}
            # where input_num is incremented in log()
            input_num = self.logger_config.input_data_logged[split]

        return (
            f"{self.input_data_path}/{split}/"
            f"data_{input_num}.{self.INPUT_DATA_FILE_EXT}"
        )

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
        meta: Union[List[str], List[int], None] = None,
        **kwargs: Any,
    ) -> None:
        """Log a dataset/iterable of input samples.

        Provide the dataset and the keys to index into it. See child for details"""

    def validate_ids_for_split(self, ids: List[int]) -> None:
        """Validate ids for the current split

        Validates:
        - that the ids are unique for the current split
        - that the ids are not already logged for the current split

        On success:
        - adds the ids to the logged_input_ids for the current split
        """
        split = self.split_name
        exc = (
            "If you've re-run a block of code or notebook cell that logs model "
            "outputs, that could be the cause. Try reinitializing with `dq.init` "
            "to clear your local environment, and then logging your data again. Call "
            "`dq.enable_galileo_verbose()` to see the duplicate IDs"
        )
        id_set = set(ids)
        if len(id_set) != len(ids):
            exc = "It seems you do not have unique ids in this logged data. " + exc
            if galileo_verbose_logging():
                dups = {k: v for k, v in Counter(ids).items() if v > 1}
                exc += f"split:{split}, dup ids and counts: {dups}"
            raise GalileoException(exc)
        # This means some logged ids were already logged!
        if len(id_set - self.logger_config.logged_input_ids[split]) != len(ids):
            exc = "Some ids in this dataset were already logged for this split. " + exc
            if galileo_verbose_logging():
                overlapping = self.logger_config.logged_input_ids[split].intersection(
                    id_set
                )
                exc += f"split:{split}, overlapping ids: {overlapping}"
            raise GalileoException(exc)

        self.logger_config.logged_input_ids[split].update(ids)

    def add_ids_to_split(self, ids: List) -> None:
        if self.split:
            self.logger_config.idx_to_id_map[str(self.split)].extend(ids)

    def log(self) -> None:
        """Writes input data to disk in .galileo/logs

        If input data already exist, append new data to existing input file.
        If the dataset is very large this function will be called multiple
        times for a given split.
        """
        self.validate_and_format()
        # E.g. /Users/username/.galileo/logs/proj-id/run-id
        write_input_dir = self.write_output_dir
        os.makedirs(write_input_dir, exist_ok=True)
        # E.g. /Users/username/.galileo/logs/proj-id/run-id/training
        os.makedirs(f"{self.input_data_path}/{self.split}", exist_ok=True)

        df = self._get_input_df()
        # Validates cloud size limit
        self.validate_data_size(df)

        ids = df["id"].tolist()
        self.validate_ids_for_split(ids)
        self.add_ids_to_split(ids)

        self.export_df(df)

    def export_df(self, df: vaex.DataFrame) -> None:
        """Export the dataframe and increment the input_data_logged
        in this helper in order to allow for overrides in child classes.

        For instance semseg needs to do this in a multithreaded way and
        add locks to avoid threading issues
        """
        file_path = self.input_data_file()
        if self.log_export_progress:
            with vaex.progress.tree("vaex", title=f"Logging {len(df)} samples"):
                df.export(file_path)
        else:
            df.export(file_path)

        df.close()
        self.logger_config.input_data_logged[str(self.split)] += 1

    @property
    def support_data_embs(self) -> bool:
        return True

    def upload(
        self, last_epoch: Optional[int] = None, create_data_embs: bool = False
    ) -> None:
        """
        Iterates through all of each splits children folders [data/emb/prob] for each
        inference name / epoch, concatenates all of the files with vaex, and uploads
        them to a single file in minio

        If create_data_embs is True, this will also run an off the shelf transformer
        and upload those text embeddings alongside the models finetuned embeddings
        """
        # For linting
        assert (
            config.current_project_id and config.current_run_id
        ), "You must call dq.init and train a model before calling finish"
        ThreadPoolManager.wait_for_threads()
        self.check_for_logging_failures()
        print("☁️ Uploading Data")
        object_store = ObjectStore()
        proj_run = f"{config.current_project_id}/{config.current_run_id}"
        location = f"{self.LOG_FILE_DIR}/{proj_run}"

        if cuml_available():
            apply_umap_to_embs(location, last_epoch)
        else:
            print(
                "CuML libraries not found, running standard process. "
                "For faster Galileo processing, consider installing\n"
                "`pip install 'dataquality[cuda]' --extra-index-url="
                "https://pypi.nvidia.com/`"
            )

        if cuml_available() and create_data_embs and self.support_data_embs:
            print("Creating and uploading data embeddings")
            upload_umap_data_embs(
                config.current_project_id,
                config.current_run_id,
                self.input_data_path,
                location,
                last_epoch,
            )
            # We have already created them here, so don't try again later
            create_data_embs = False

        for split in Split.get_valid_attributes():
            self.upload_split(
                location, split, object_store, last_epoch, create_data_embs
            )

    def upload_split(
        self,
        location: str,
        split: str,
        object_store: ObjectStore,
        last_epoch: Optional[int],
        create_data_embs: bool,
    ) -> None:
        split_loc = f"{location}/{split}"
        in_frame_path = f"{self.input_data_path}/{split}"
        input_logged = os.path.exists(in_frame_path)
        output_logged = os.path.exists(split_loc)
        if not output_logged:
            return
        if not input_logged:
            warnings.warn(
                f"There was output data logged for split {split} but no input data "
                "logged. Skipping upload for this split as there are no samples "
                "to join to.",
                GalileoWarning,
            )
            return
        in_frame_split = vaex.open(f"{in_frame_path}/*.{self.INPUT_DATA_FILE_EXT}")
        in_frame_split = self.convert_large_string(in_frame_split)
        self.upload_split_from_in_frame(
            object_store,
            in_frame_split,
            split,
            split_loc,
            last_epoch,
            create_data_embs,
        )
        in_frame_split.close()
        _shutil_rmtree_retry(in_frame_path)
        gc.collect()

    @classmethod
    def create_and_upload_data_embs(
        cls, df: DataFrame, split: str, epoch_or_inf: str
    ) -> None:
        """Uploads off the shelf data embeddings for a split"""
        object_store = ObjectStore()
        df_copy = df.copy()
        try:
            data_embs = create_data_embs_df(df_copy)
        except HfHubHTTPError as e:
            warnings.warn(
                "Unable to download transformer from huggingface. Data embeddings "
                f"will be skipped. {str(e)}"
            )
            return
        proj_run_split = f"{config.current_project_id}/{config.current_run_id}/{split}"
        minio_file = f"{proj_run_split}/{epoch_or_inf}/{DATA_EMB_PATH}"
        # And upload
        object_store.create_project_run_object_from_df(data_embs, minio_file)

    def convert_large_string(self, df: DataFrame) -> DataFrame:
        """Cast regular string to large_string for the text column

        Arrow strings have a max size of 2GB, so in order to export to hdf5 and
        join the strings in the text column, we upcast to a large string.

        We only do this for types that write to HDF5 files
        """
        df_copy = df.copy()
        if "text" not in df_copy.get_column_names():
            return df_copy
        # Characters are each 1 byte. If more bytes > max, it needs to be large_string
        text_bytes = df_copy["text"].str.len().sum()
        if text_bytes > self.STRING_MAX_SIZE_B:
            df_copy["text"] = df_copy['astype(text, "large_string")']
        return df_copy

    @classmethod
    def upload_split_from_in_frame(
        cls,
        object_store: ObjectStore,
        in_frame: DataFrame,
        split: str,
        split_loc: str,
        last_epoch: Optional[int] = None,
        create_data_embs: bool = False,
    ) -> None:
        epochs_or_infs = os.listdir(split_loc)
        epochs_or_infs = sorted(
            epochs_or_infs, key=lambda i: int(i) if split != Split.inference else i
        )
        # last_epoch is inclusive
        last_epoch = last_epoch + 1 if last_epoch else last_epoch
        # If set, last_epoch will only let you upload to and including the provided
        # epoch value, nothing more.
        # If None, then slicing a list [:None] will include all values
        epochs_or_infs = epochs_or_infs[:last_epoch]

        largest_epoch = epochs_or_infs[-1]

        # For each inference name or epoch of the given split
        for epoch_or_inf in tqdm(
            epochs_or_infs,
            total=len(epochs_or_infs),
            desc=split,
            file=sys.stdout,
        ):
            input_batch = in_frame.copy()
            prob_only = cls.prob_only(epochs_or_infs, split, epoch_or_inf, last_epoch)
            if split == Split.inference:
                input_batch = filter_df(input_batch, "inference_name", epoch_or_inf)
                if not len(input_batch):
                    warnings.warn(
                        "There was output data logged for inference_name "
                        f"{epoch_or_inf} but no input data logged. Skipping upload for "
                        "this inference run as there are no samples to join to.",
                        GalileoWarning,
                    )
                    continue
            if create_data_embs and (
                split == Split.inference or epoch_or_inf == largest_epoch
            ):
                name = f"{split}/{epoch_or_inf}" if split == Split.inference else split
                print(f"Creating and uploading data embeddings for {name}")
                cls.create_and_upload_data_embs(input_batch, split, epoch_or_inf)

            dir_name = f"{split_loc}/{epoch_or_inf}"
            in_out_frames = cls.create_in_out_frames(
                input_batch, dir_name, prob_only, split, epoch_or_inf
            )
            cls.upload_in_out_frames(object_store, in_out_frames, split, epoch_or_inf)

    @classmethod
    def create_in_out_frames(
        cls,
        in_frame: DataFrame,
        dir_name: str,
        prob_only: bool,
        split: str,
        epoch_or_inf: Union[str, int],
    ) -> BaseLoggerDataFrames:
        """Formats the input data and model output data

        In this step, we concatenate the many hdf5 files created during model training
        and logging. We log those in threaded processes, and here we combine them
        into a single hdf5 file that vaex can read into a dataframe

        :param in_frame: the input dataframe
        :param dir_name: The directory of all of the output hdf5 files
        :param prob_only: If we are only uploading probability data. We only upload
            probability data for all epochs except the last one (we dont use cross-epoch
            embeddings currently, so we dont log them)
        :param split: The split we are logging for
        :param epoch_or_inf: The epoch or inference name we are logging for
        """
        out_frame = get_output_df(dir_name, prob_only, split, epoch_or_inf)
        epoch_or_inf_name = "inference_name" if split == Split.inference else "epoch"
        return cls.process_in_out_frames(
            in_frame, out_frame, prob_only, epoch_or_inf_name, split
        )

    @classmethod
    def process_in_out_frames(
        cls,
        in_frame: DataFrame,
        out_frame: DataFrame,
        prob_only: bool,
        epoch_or_inf_name: str,
        split: str,
    ) -> BaseLoggerDataFrames:
        """Processes input and output dataframes from logging

        Validates uniqueness of IDs in the output dataframe
        Joins inputs and outputs
        Splits the dataframes into prob, emb, and data for uploading to minio

        :param in_frame: The input dataframe
        :param out_frame: The model output dataframe
        :param prob_only: If we are only uploading probabilities, or everything
        :param epoch_or_inf_name: The epoch or inference name we are uploading for
        """
        validate_unique_ids(out_frame, epoch_or_inf_name)
        allow_missing_in_df_ids = cls.logger_config.dataloader_random_sampling

        in_out = _join_in_out_frames(
            in_frame, out_frame, allow_missing_in_df_ids=allow_missing_in_df_ids
        )

        dataframes = cls.separate_dataframe(in_out, prob_only, split)
        # These df vars will be used in upload_in_out_frames
        dataframes.emb.set_variable("skip_upload", prob_only)
        dataframes.data.set_variable("skip_upload", prob_only)
        epoch_inf_val = out_frame[[epoch_or_inf_name]][0][0]
        dataframes.prob.set_variable("progress_name", str(epoch_inf_val))

        return dataframes

    @classmethod
    def upload_in_out_frames(
        cls,
        object_store: ObjectStore,
        in_out_frames: BaseLoggerDataFrames,
        split: str,
        epoch_or_inf: Union[str, int],
    ) -> None:
        proj_run = f"{config.current_project_id}/{config.current_run_id}"

        prob = in_out_frames.prob
        emb = in_out_frames.emb
        data_df = in_out_frames.data

        epoch_inf = prob.variables.pop(DFVar.progress_name, "")

        name = "inf_name" if split == Split.inference else "epoch"
        desc = f"{split} ({name}={epoch_inf})"

        for data_folder, df_obj in tqdm(
            zip(DATA_FOLDERS, [emb, prob, data_df]),
            total=3,
            desc=desc,
            leave=False,
            file=sys.stdout,
        ):
            if df_obj.variables.get(DFVar.skip_upload):
                continue
            ext = cls.DATA_FOLDER_EXTENSION[data_folder]
            minio_file = (
                f"{proj_run}/{split}/{epoch_or_inf}/{data_folder}/{data_folder}.{ext}"
            )
            cls._handle_numpy_types(df=df_obj)
            object_store.create_project_run_object_from_df(
                df=df_obj, object_name=minio_file
            )

    @classmethod
    def _handle_numpy_types(cls, df: DataFrame) -> None:
        """Casts everything to float/int 32"""
        for col, dt in zip(df.get_column_names(), df.dtypes):
            if dt == "float" and dt != "float32":
                df[col] = df[col].astype("float32")
            elif dt == "int" and dt != "int32":
                df[col] = df[col].astype("int32")

    @classmethod
    def prob_only(
        cls,
        epochs: List[str],
        split: str,
        epoch_or_inf_name: Union[int, str],
        last_epoch: Optional[int],
    ) -> bool:
        """Determines if we are only uploading probabilities

        For all epochs that aren't the last 2 (early stopping), we only want to
        upload the probabilities (for DEP calculation).
        """
        if split == Split.inference:  # Inference doesn't have DEP
            return False

        # If split is not inference, epoch_or_inf must be epoch
        epoch = int(epoch_or_inf_name)
        max_epoch_for_split = max([int(i) for i in epochs])
        if last_epoch is not None:
            max_epoch_for_split = min(max_epoch_for_split, last_epoch)
        return bool(epoch < max_epoch_for_split - 1)

    def validate_and_format(self) -> None:
        """Validates the logger

        Ensures that self.split is set, or sets it to the current split
        from the logger_config.

        Each child also defines an additional validate method that is called
        """
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
        valid_meta_cols = []
        for key, values in list(self.meta.items()):
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
            invalid_values = filter(lambda t: not isinstance(t, valid_types), values)
            bad_val = next(invalid_values, None)
            if bad_val:
                warnings.warn(
                    f"Metadata column {key} has one or more invalid values {bad_val} "
                    f"of type {type(bad_val)}.",
                    GalileoWarning,
                )
                continue
            valid_meta_cols.append(key)

        def valid_str_col(df: DataFrame, key: str) -> bool:
            """Valid str col checks length of longest str in metadata col"""
            if df[key].dtype != "string":
                return True

            max_str_len = df[key].str.len().max()
            if max_str_len > self.MAX_DOC_LEN:
                warnings.warn(
                    f"Metadata column {key} has one or more strings that are longer "
                    f"than max document length of {self.MAX_DOC_LEN} characters. "
                    "Will not log this metadata column.",
                    GalileoWarning,
                )
                return False
            if max_str_len > self.MAX_STR_LEN:
                if len(self.logger_config.metadata_documents) >= self.LIMIT_NUM_DOCS:
                    warnings.warn(
                        "You have already logged limit of 3 document columns. A "
                        "document column is a column that has max str length between"
                        f"1,000 and 10,000 characters. Metadata column {key} has one "
                        f"or more strings that are longer than {self.MAX_STR_LEN} "
                        "characters. Will not log this metadata column.",
                        GalileoWarning,
                    )
                    return False
                else:
                    self.logger_config.metadata_documents.add(key)

            return True

        df: DataFrame = vaex.from_dict(
            {k: v for k, v in self.meta.items() if k in valid_meta_cols}
        )
        valid_meta_cols = [k for k in valid_meta_cols if valid_str_col(df, k)]
        valid_meta_cols = valid_meta_cols[: self.MAX_META_COLS]  # Take first 25
        self.meta = {k: v for k, v in self.meta.items() if k in valid_meta_cols}

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
    def separate_dataframe(
        cls, df: DataFrame, prob_only: bool = False, split: Optional[str] = None
    ) -> BaseLoggerDataFrames:
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

    def validate_data_size(self, df: DataFrame) -> None:
        """Validates that the data size is within the limits of Galileo Cloud

        If the data size is too large, a warning is raised.
        """
        if not is_galileo_cloud():
            return
        samples_logged = len(df)
        path_to_logged_data = f"{self.input_data_path}/*/*arrow"
        if glob.glob(path_to_logged_data):
            samples_logged += len(vaex.open(f"{self.input_data_path}/*/*arrow"))
        nrows = BaseGalileoDataLogger.MAX_DATA_SIZE_CLOUD
        if samples_logged > nrows:
            warnings.warn(
                f"⚠️ Hey there! You've logged over {nrows} rows in your input data. "
                f"Galileo Cloud only supports up to {nrows} rows. "
                "If you are using larger datasets, you may see degraded performance. "
                "Please email us at team@rungalileo.io if you have any questions.",
                GalileoWarning,
            )
