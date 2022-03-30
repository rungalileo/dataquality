from enum import Enum, unique
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import vaex
from vaex.dataframe import DataFrame

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
    inference_name = "inference_name"

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
    * split: The split for training/test/validation
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
        inference_name: str = None,
    ) -> None:
        """Create data logger.

        :param text: The raw text inputs for model training. List[str]
        :param labels: the ground truth labels aligned to each text field.
        List[str]
        :param ids: Optional unique indexes for each record. If not provided, will
        default to the index of the record. Optional[List[Union[int,str]]]
        :param split: The split for training/test/validation
        """
        super().__init__(meta)
        # Need to compare to None because they may be np arrays which cannot be
        # evaluated with bool directly
        self.text = text if text is not None else []
        self.labels = [str(i) for i in labels] if labels is not None else []
        self.ids = ids if ids is not None else []
        self.split = split
        self.inference_name = inference_name

    @staticmethod
    def get_valid_attributes() -> List[str]:
        """
        Returns a list of valid attributes that this logger accepts
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
        super().validate()

        label_len = len(self.labels)
        text_len = len(self.text)
        id_len = len(self.ids)

        self.text = list(self._convert_tensor_ndarray(self.text))
        self.labels = list(self._convert_tensor_ndarray(self.labels, attr="Labels"))
        self.ids = list(self._convert_tensor_ndarray(self.ids))

        if self.split == Split.inference.value:
            assert not label_len, "You cannot have labels in your inference split!"
            assert self.inference_name, (
                "Inference name must be set when logging an inference split. Use "
                "set_split('inference', inference_name) to set inference name"
            )

        else:
            assert label_len and text_len, (
                f"Both text and labels for your logger must be set, but got"
                f" text:{bool(text_len)}, labels:{bool(label_len)}"
            )

            assert text_len == label_len, (
                f"labels and text must be the same length, but got"
                f"(labels, text) ({label_len}, {text_len})"
            )

        if self.ids:
            assert id_len == text_len, (
                f"Ids exists but are not the same length as text and labels. "
                f"(ids, text) ({id_len}, {text_len})"
            )
        else:
            self.ids = list(range(text_len))

        self.validate_metadata(batch_size=text_len)

    def _get_input_df(self) -> DataFrame:
        inp = dict(
            id=self.ids,
            text=self.text,
            split=self.split,
            data_schema_version=__data_schema_version__,
            gold=self.labels if self.split != Split.inference.value else None,
            **self.meta,
        )
        if self.inference_name:
            inp.update(inference_name=self.inference_name)
        return vaex.from_pandas(pd.DataFrame(inp))

    @classmethod
    def split_dataframe(
        cls, df: DataFrame, prob_only: bool
    ) -> Tuple[DataFrame, DataFrame, DataFrame]:
        """Splits the singular dataframe into its 3 components

        Gets the probability df, the embedding df, and the "data" df containing
        all other columns
        """
        df_copy = df.copy()
        # Separate out embeddings and probabilities into their own files
        prob_cols = cls._get_prob_cols()
        prob = df_copy[prob_cols]

        if prob_only:  # In this case, we don't care about the other columns
            emb_cols = ["id"]
            other_cols = ["id"]
        else:
            emb_cols = ["id", "emb"]
            ignore_cols = ["emb", "split_id"] + prob_cols
            other_cols = [i for i in df_copy.get_column_names() if i not in ignore_cols]
            other_cols += ["id"]

        emb = df_copy[emb_cols]
        data_df = df_copy[other_cols]
        return prob, emb, data_df

    @classmethod
    def _get_prob_cols(cls) -> List[str]:
        return ["id", "prob", "gold"]

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
