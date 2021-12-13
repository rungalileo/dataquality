import os
from glob import glob
from uuid import uuid4

import pandas as pd
import vaex

from dataquality import config
from dataquality.clients import object_store
from dataquality.loggers import BaseGalileoLogger
from dataquality.loggers.data_logger.base_data_logger import BaseGalileoDataLogger
from dataquality.loggers.data_logger.text_classification import (
    TextClassificationDataLogger,
)
from dataquality.schemas import __data_schema_version__
from dataquality.schemas.split import Split
from dataquality.utils.thread_pool import ThreadPoolManager
from dataquality.utils.vaex import _join_in_out_frames, _validate_unique_ids

DATA_FOLDERS = ["emb", "prob", "data"]


class TextMultiLabelDataLogger(TextClassificationDataLogger):
    """
    Class for logging input data/metadata of Text Multi Label models to Galileo.

    * text: The raw text inputs for model training. List[str]
    * labels: the ground truth labels aligned to each text field. List[Union[str,int]]
    * ids: Optional unique indexes for each record. If not provided, will default to
    the index of the record. Optional[List[Union[int,str]]]
    """

    __logger_name__ = "text_multi_label"

    def validate(self) -> None:
        """
        Parent validation (text_classification) with additional validation on labels

        in multi_label modeling, each element in self.labels should itself be a list
        """
        super().validate()
        for i in self.labels:
            assert isinstance(
                i, list
            ), "labels must be a list of lists in multi-label tasks"

    def log(self) -> None:
        self.validate()
        write_input_dir = (
            f"{BaseGalileoLogger.LOG_FILE_DIR}/{config.current_project_id}/"
            f"{config.current_run_id}"
        )
        if not os.path.exists(write_input_dir):
            os.makedirs(write_input_dir)
        inp = dict(
            id=self.ids,
            text=self.text,
            split=self.split,
            data_schema_version=__data_schema_version__,
            gold=self.labels if self.split != Split.inference.value else None,
            **self.meta,
        )
        df = vaex.from_pandas(pd.DataFrame(inp))
        file_path = f"{write_input_dir}/{BaseGalileoDataLogger.INPUT_DATA_NAME}"
        if os.path.isfile(file_path):
            new_name = f"{write_input_dir}/{str(uuid4()).replace('-', '')[:12]}.arrow"
            os.rename(file_path, new_name)
            vaex.concat([df, vaex.open(new_name)]).export(file_path)
            os.remove(new_name)
        else:
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

                # Separate out embeddings and probabilities into their own files
                prob = in_out[["id", "prob", "gold"]]
                emb = in_out[["id", "emb"]]
                ignore_cols = ["emb", "prob", "gold", "split_id"]
                other_cols = [
                    i for i in in_out.get_column_names() if i not in ignore_cols
                ]
                in_out = in_out[other_cols]

                for data_folder, df_obj in zip(DATA_FOLDERS, [emb, prob, in_out]):
                    minio_file = (
                        f"{proj_run}/{split}/{epoch}/{data_folder}/{data_folder}.hdf5"
                    )
                    object_store.create_project_run_object_from_df(df_obj, minio_file)
