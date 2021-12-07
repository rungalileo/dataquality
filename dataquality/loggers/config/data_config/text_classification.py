import os
import shutil
from enum import Enum, unique
from glob import glob
from typing import List, Union, Dict
from uuid import uuid4

import pandas as pd
import vaex

from dataquality import config
from dataquality.clients import object_store
from dataquality.loggers.config.base_config import BaseGalileoConfig
from dataquality.loggers.config.data_config import BaseGalileoDataConfig
from dataquality.schemas import __data_schema_version__
from dataquality.schemas.split import Split
from dataquality.utils.thread_pool import ThreadPoolManager
from dataquality.utils.vaex import _validate_unique_ids, _join_in_out_frames

MAX_META_COLS = BaseGalileoDataConfig.MAX_META_COLS
MAX_STR_LEN = BaseGalileoDataConfig.MAX_STR_LEN
DATA_FOLDERS = ["emb", "prob", "data"]


@unique
class GalileoDataConfigAttributes(str, Enum):
    text = "text"
    labels = "labels"
    ids = "ids"
    # mixin restriction on str (due to "str".split(...))
    split = "split"  # type: ignore
    meta = "meta"  # Metadata columns for logging

    @staticmethod
    def get_valid() -> List[str]:
        return list(map(lambda x: x.value, GalileoDataConfigAttributes))


class TextClassificationDataConfig(BaseGalileoDataConfig):
    """
    Class for logging input data/metadata of Text Classification models to Galileo.

    * text: The raw text inputs for model training. List[str]
    * labels: the ground truth labels aligned to each text field. List[Union[str,int]]
    * ids: Optional unique indexes for each record. If not provided, will default to
    the index of the record. Optional[List[Union[int,str]]]
    """
    __name__ = "text_classification"

    def __init__(
        self,
        text: List[str] = None,
        labels: List[str] = None,
        ids: List[Union[int, str]] = None,
        split: str = None,
        **kwargs: Dict[str, List[Union[str, float, int]]],
    ) -> None:
        super().__init__()
        # Need to compare to None because they may be np arrays which cannot be
        # evaluated with bool directly
        self.text = text if text is not None else []
        self.labels = [str(i) for i in labels] if labels is not None else []
        self.ids = ids if ids is not None else []
        self.split = split
        self.meta = kwargs

    @staticmethod
    def get_valid_attributes() -> List[str]:
        """
        Returns a list of valid attributes that GalileoModelConfig accepts
        :return: List[str]
        """
        return GalileoDataConfigAttributes.get_valid()

    def validate(self) -> None:
        """
        Validates that the current config is correct.
        * Text and Labels must both exist (unless split is 'inference' in which case
        labels must be None)
        * Text and Labels must be the same length
        * If ids exist, it must be the same length as text/labels
        :return: None
        """

        label_len = len(self.labels)
        text_len = len(self.text)
        id_len = len(self.ids)

        self.text = self._convert_tensor_ndarray(self.text)
        self.labels = self._convert_tensor_ndarray(self.labels)
        self.ids = self._convert_tensor_ndarray(self.ids)

        assert self.split, "Your GalileoDataConfig has no split!"
        self.split = Split.training.value if self.split == "train" else self.split
        self.split = self.split.value if isinstance(self.split, Split) else self.split
        assert (
            isinstance(self.split, str) and self.split in Split.get_valid_attributes()
        ), (
            f"Split should be one of {Split.get_valid_attributes()} "
            f"but got {self.split}"
        )
        if self.split == Split.inference.value:
            assert not len(
                self.labels
            ), "You cannot have labels in your inference split!"
        else:
            assert label_len and text_len, (
                f"Both text and labels for your GalileoDataConfig must be set, but got"
                f" text:{bool(text_len)}, labels:{bool(text_len)}"
            )

            assert text_len == label_len, (
                f"labels and text must be the same length, but got"
                f"(labels, text) ({label_len},{text_len})"
            )

        if self.ids:
            assert id_len == text_len, (
                f"Ids exists but are not the same length as text and labels. "
                f"(ids, text) ({id_len}, {text_len})"
            )
        else:
            self.ids = list(range(text_len))

        self.validate_metadata(batch_size=text_len)

    def log(self) -> None:
        self.validate()
        write_input_dir = (
            f"{BaseGalileoConfig.LOG_FILE_DIR}/{config.current_project_id}/"
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
        file_path = f"{write_input_dir}/{BaseGalileoDataConfig.INPUT_DATA_NAME}"
        if os.path.isfile(file_path):
            new_name = f"{write_input_dir}/{str(uuid4()).replace('-', '')[:12]}.arrow"
            os.rename(file_path, new_name)
            vaex.concat([df, vaex.open(new_name)]).export(file_path)
            os.remove(new_name)
        else:
            df.export(file_path)
        df.close()

    def upload(self):
        """
        Iterates through all of the splits/epochs/[data/emb/prob] folders, concatenates
        all of the files with vaex, and uploads them to a single file in minio in the same
        directory structure
        """
        ThreadPoolManager.wait_for_threads()
        print("☁️ Uploading Data")
        proj_run = f"{config.current_project_id}/{config.current_run_id}"
        location = f"{self.LOG_FILE_DIR}/{proj_run}"

        in_frame = vaex.open(f"{location}/{BaseGalileoDataConfig.INPUT_DATA_NAME}").copy()
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
                other_cols = [i for i in in_out.get_column_names() if
                              i not in ignore_cols]
                in_out = in_out[other_cols]

                for data_folder, df_obj in zip(DATA_FOLDERS, [emb, prob, in_out]):
                    minio_file = (
                        f"{proj_run}/{split}/{epoch}/{data_folder}/{data_folder}.hdf5"
                    )
                    object_store.create_project_run_object_from_df(df_obj, minio_file)

