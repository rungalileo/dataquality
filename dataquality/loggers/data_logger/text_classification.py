import os
from enum import Enum, unique
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import uuid4

import pandas as pd
import vaex
from vaex.dataframe import DataFrame

from dataquality.core._config import config
from dataquality.loggers import BaseGalileoLogger
from dataquality.loggers.data_logger.base_data_logger import BaseGalileoDataLogger
from dataquality.loggers.logger_config.text_classification import (
    text_classification_logger_config,
)
from dataquality.schemas import __data_schema_version__
from dataquality.schemas.split import Split


@unique
class GalileoDataLoggerAttributes(str, Enum):
    text = "text"
    labels = "labels"
    ids = "ids"
    # mixin restriction on str (due to "str".split(...))
    split = "split"  # type: ignore
    meta = "meta"  # Metadata columns for logging

    @staticmethod
    def get_valid() -> List[str]:
        return list(map(lambda x: x.value, GalileoDataLoggerAttributes))


class TextClassificationDataLogger(BaseGalileoDataLogger):
    """
    Class for logging input data/metadata of Text Classification models to Galileo.

    * text: The raw text inputs for model training. List[str]
    * labels: the ground truth labels aligned to each text field. List[str]
    * ids: Optional unique indexes for each record. If not provided, will default to
    the index of the record. Optional[List[int]]
    """

    __logger_name__ = "text_classification"
    logger_config = text_classification_logger_config

    def __init__(
        self,
        text: List[str] = None,
        labels: List[str] = None,
        ids: List[int] = None,
        split: str = None,
        meta: Optional[Dict[str, List[Union[str, float, int]]]] = None,
    ) -> None:
        """Create data logger.

        :param text: The raw text inputs for model training. List[str]
        :param labels: the ground truth labels aligned to each text field.
        List[str]
        :param ids: Optional unique indexes for each record. If not provided, will
        default to the index of the record. Optional[List[Union[int,str]]]
        """
        super().__init__(meta)
        # Need to compare to None because they may be np arrays which cannot be
        # evaluated with bool directly
        self.text = text if text is not None else []
        self.labels = [str(i) for i in labels] if labels is not None else []
        self.ids = ids if ids is not None else []
        self.split = split

    @staticmethod
    def get_valid_attributes() -> List[str]:
        """
        Returns a list of valid attributes that GalileoModelConfig accepts
        :return: List[str]
        """
        return GalileoDataLoggerAttributes.get_valid()

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

        self.text = list(self._convert_tensor_ndarray(self.text))
        self.labels = list(self._convert_tensor_ndarray(self.labels, attr="Labels"))
        self.ids = list(self._convert_tensor_ndarray(self.ids))

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
            assert not label_len, "You cannot have labels in your inference split!"
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
            f"{BaseGalileoLogger.LOG_FILE_DIR}/{config.current_project_id}/"
            f"{config.current_run_id}"
        )
        if not os.path.exists(write_input_dir):
            os.makedirs(write_input_dir)
        inp = self._get_input_dict()
        df = vaex.from_pandas(pd.DataFrame(inp))
        file_path = f"{write_input_dir}/{BaseGalileoDataLogger.INPUT_DATA_NAME}"
        if os.path.isfile(file_path):
            new_name = f"{write_input_dir}/{str(uuid4()).replace('-', '')[:12]}.arrow"
            os.rename(file_path, new_name)
            vaex.concat([df, vaex.open(new_name)]).export(file_path, progress="vaex")
            os.remove(new_name)
        else:
            df.export(file_path, progress="vaex")
        df.close()

    def _get_input_dict(self) -> Dict[str, Any]:
        return dict(
            id=self.ids,
            text=self.text,
            split=self.split,
            data_schema_version=__data_schema_version__,
            gold=self.labels if self.split != Split.inference.value else None,
            **self.meta,
        )

    @classmethod
    def split_dataframe(cls, df: DataFrame) -> Tuple[DataFrame, DataFrame, DataFrame]:
        """Splits the singular dataframe into its 3 components

        Gets the probability df, the embedding df, and the "data" df containing
        all other columns
        """
        df_copy = df.copy()
        # Separate out embeddings and probabilities into their own files
        prob = df_copy[["id", "prob", "gold"]]
        emb = df_copy[["id", "emb"]]
        ignore_cols = ["emb", "prob", "gold", "split_id"]
        other_cols = [i for i in df_copy.get_column_names() if i not in ignore_cols]
        data_df = df_copy[other_cols]
        return prob, emb, data_df

    @classmethod
    def validate_labels(cls) -> None:
        assert cls.logger_config.labels, (
            "You must set your config labels before calling finish. "
            "See `dataquality.set_labels_for_run`"
        )

        assert len(cls.logger_config.labels) == cls.logger_config.observed_num_labels, (
            f"You set your labels to be {cls.logger_config.labels} "
            f"({len(cls.logger_config.labels)} labels) but based on training, your "
            f"model is expecting {cls.logger_config.observed_num_labels} labels. "
            f"Use dataquality.set_labels_for_run to update your config labels."
        )
