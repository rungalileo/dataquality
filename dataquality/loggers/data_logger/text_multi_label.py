from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import vaex
from vaex.dataframe import DataFrame

from dataquality.exceptions import GalileoException
from dataquality.loggers.data_logger.base_data_logger import (
    BaseGalileoDataLogger, MetasType, MetaType)
from dataquality.loggers.logger_config.text_multi_label import (
    TextMultiLabelLoggerConfig, text_multi_label_logger_config)
from dataquality.schemas import __data_schema_version__
from dataquality.schemas.dataframe import BaseLoggerDataFrames
from dataquality.schemas.split import Split

DATA_FOLDERS = ["emb", "prob", "data"]


class TextMultiLabelDataLogger(BaseGalileoDataLogger):
    __logger_name__ = "text_multi_label"
    logger_config: TextMultiLabelLoggerConfig = text_multi_label_logger_config

    DATA_FOLDER_EXTENSION = {"emb": "hdf5", "prob": "arrow", "data": "arrow"}

    def __init__(
        self,
        split: Optional[str] = None,
        ids: Optional[List[int]] = None,
        texts: Optional[List[str]] = None,
        labels: Optional[List[List[str]]] = None,
        active_labels: Optional[List[List[int]]] = None,
        meta: Optional[MetasType] = None,
    ) -> None:
        super().__init__(meta=meta)
        self.split = split
        self.ids = ids or []
        self.texts = texts or []
        self.labels = labels or []
        self.active_labels = active_labels or []

    def log_data_sample(
        self,
        *,
        text: str,
        id: int,
        split: Optional[Split] = None,
        label: Optional[str] = None,
        labels: Optional[List[str]] = None,
        active_labels: Optional[List[int]] = None,
        meta: Optional[MetaType] = None,
        inference_name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.validate_kwargs(kwargs)
        if label:
            raise GalileoException("In multi-label, use labels instead of label.")
        self.ids = [id]
        self.texts = [text]
        self.split = split
        self.labels = [[str(i) for i in labels]] if labels else []
        self.active_labels = [active_labels] if active_labels else []
        self.inference_name = inference_name
        self.meta = {i: [meta[i]] for i in meta} if meta else {}
        self.log()

    def log_data_samples(
        self,
        *,
        texts: List[str],
        ids: List[int],
        labels: Optional[List[List[str]]] = None,
        active_labels: Optional[List[List[int]]] = None,
        split: Optional[Split] = None,
        inference_name: Optional[str] = None,
        meta: Optional[MetasType] = None,
        **kwargs: Any,
    ) -> None:
        self.validate_kwargs(kwargs)
        self.ids = ids
        self.texts = texts
        self.split = split
        self.labels = labels if labels is not None else []
        self.inference_name = inference_name
        self.active_labels = active_labels if active_labels is not None else []
        self.meta = meta or {}
        self.active_labels = self._set_active_labels(self.labels)
        self.log()

    def _set_active_labels(self, labels: List[List[str]]) -> List[List[int]]:
        _labels_arr = self.logger_config.labels
        assert _labels_arr is not None, "You must set labels before logging input data"
        result = []
        for _labels in labels:
            result.append(self._set_active_label_values(_labels, _labels_arr))
        return result

    def _set_active_label_values(
        self, labels: List[str], _labels_arr: List[str]
    ) -> List[int]:
        result = np.zeros(len(_labels_arr), dtype=int)
        for label in labels:
            result[_labels_arr.index(label)] = 1
        return result.tolist()

    def _get_input_df(self) -> DataFrame:
        df_len = len(self.texts)
        inp = dict(
            id=self.ids,
            text=self.texts,
            split=[Split(self.split).value] * df_len,
            data_schema_version=[__data_schema_version__] * df_len,
            gold=pa.array(self.labels) if self.split != Split.inference.value else None,
            active_labels=pa.array(self.active_labels)
            if self.split != Split.inference.value
            else None,
            **self.meta,
        )
        if self.inference_name:
            inp.update(inference_name=self.inference_name)
        return vaex.from_dict(inp)

    def _log_dict(
        self,
        _dict: Dict,
        meta: Dict,
        split: Optional[Split] = None,
        inference_name: Optional[str] = None,
    ) -> None:
        self.log_data_samples(
            texts=_dict["text"],
            ids=_dict["id"],
            labels=_dict["label"],
            active_labels=_dict["active_labels"],
            split=split,
            inference_name=inference_name,
            meta=meta,
        )

    def validate_and_format(self) -> None:
        super().validate_and_format()
        self.logger_config.observed_num_labels = len(self.active_labels[0])
        label_len = len(self.labels)
        text_len = len(self.texts)
        id_len = len(self.ids)

        set_labels_are_ints = self.logger_config.int_labels

        if self.split != Split.inference and str(self.labels[0]).isnumeric():
            # labels must be set if numeric
            assert self.logger_config.labels is not None, (
                "You must set labels before logging input data,"
                " when label column is numeric"
            )

        if label_len and isinstance(self.labels[0], int) and not set_labels_are_ints:
            self.labels = [self.logger_config.labels[lbl] for lbl in self.labels]

        if not isinstance(self.texts, list):
            self.texts = list(self._convert_tensor_ndarray(self.texts))

        clean_labels = self._convert_tensor_ndarray(self.labels, attr="Labels")
        # If the dtype if object, we have a ragged nested sequence, so we need to
        # iterate list by list converting to strings
        if clean_labels.dtype == object:
            self.labels = [np.array(i).astype("str").tolist() for i in clean_labels]
        # Normal nparray, can convert to string elements directly
        else:
            self.labels = clean_labels.astype("str").tolist()

        self.ids = list(self._convert_tensor_ndarray(self.ids))

        if self.split == Split.inference.value:
            assert not label_len, "You cannot have labels in your inference split!"
            if not self.inference_name:
                self.inference_name = self.logger_config.cur_inference_name
            assert self.inference_name, (
                "Inference name must be set when logging an inference split. Use "
                "set_split('inference', inference_name) to set inference name"
            )

        else:
            assert label_len and text_len, (
                f"You must log both text and labels for split {self.split}."
                f" Text samples logged:{text_len}, labels logged:{label_len}"
            )

            assert text_len == label_len, (
                f"labels and text must be the same length, but got"
                f"(labels, text) ({label_len}, {text_len})"
            )

        assert id_len == text_len, (
            f"Ids exists but are not the same length as text and labels. "
            f"(ids, text) ({id_len}, {text_len})"
        )

        self.validate_logged_labels()
        self.validate_metadata(batch_size=text_len)

    def validate_logged_labels(self) -> None:
        for input_labels in self.labels:
            assert isinstance(
                input_labels, (list, np.ndarray, pd.Series, pa.lib.StringArray)
            ), f"labels must be a list of lists in multi-label tasks, but got {type(input_labels)}"

    @classmethod
    def validate_labels(cls) -> None:
        assert cls.logger_config.labels, (
            "You must set your config labels before calling finish. "
            "See `dataquality.set_labels_for_run`"
        )

    @classmethod
    def validate_observed_num_labels(cls) -> None:
        assert cls.logger_config.observed_num_labels, (
            "observed_num_labels must be set before calling finish. "
            "See `dataquality.set_labels_for_run`"
        )

    @classmethod
    def _get_prob_cols(cls) -> List[str]:
        return ["id", "prob", "active_labels"]

    @classmethod
    def separate_dataframe(
        cls, df: DataFrame, prob_only: bool = True, split: Optional[str] = None
    ) -> BaseLoggerDataFrames:
        """Separates the singular dataframe into its 3 components

        Gets the probability df, the embedding df, and the "data" df containing
        all other columns
        """
        df_copy = df.copy()
        # Separate out embeddings and probabilities into their own files

        prob_cols = cls._get_prob_cols()
        prob = df_copy[prob_cols]
        prob["prob"] = pa.array(prob["prob"].tolist())

        if prob_only:  # In this case, we don't care about the other columns
            emb_cols = ["id"]
            other_cols = ["id"]
        else:
            emb_cols = ["id", "emb", "x", "y", "emb_pca"]
            emb_cols = [c for c in emb_cols if c in df_copy.get_column_names()]
            ignore_cols = ["split_id"] + prob_cols + emb_cols
            other_cols = [i for i in df_copy.get_column_names() if i not in ignore_cols]
            other_cols += ["id"]

        emb = df_copy[emb_cols]
        data_df = df_copy[other_cols]
        data_df["pred"] = pa.array(data_df["pred"].tolist())
        return BaseLoggerDataFrames(prob=prob, emb=emb, data=data_df)
