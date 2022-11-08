from collections import defaultdict
from enum import Enum, unique
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import vaex
from vaex.dataframe import DataFrame

from dataquality.exceptions import GalileoException
from dataquality.loggers.data_logger.base_data_logger import (
    ITER_CHUNK_SIZE,
    BaseGalileoDataLogger,
    DataSet,
    MetasType,
    MetaType,
)
from dataquality.loggers.logger_config.text_classification import (
    text_classification_logger_config,
)
from dataquality.schemas import __data_schema_version__
from dataquality.schemas.split import Split
from dataquality.utils.vaex import rename_df


@unique
class GalileoDataLoggerAttributes(str, Enum):
    texts = "texts"
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

    * texts: The raw text inputs for model training. List[str]
    * labels: the ground truth labels aligned to each text field. List[str]
    * ids: Optional unique indexes for each record. If not provided, will default to
    the index of the record. Optional[List[int]]
    * split: The split for training/test/validation

    ex:
    .. code-block:: python

        all_labels = ["A", "B", "C"]
        dq.set_labels_for_run(labels = all_labels)

        texts: List[str] = [
            "Text sample 1",
            "Text sample 2",
            "Text sample 3",
            "Text sample 4"
        ]

        labels: List[str] = ["B", "C", "A", "A"]

        ids: List[int] = [0, 1, 2, 3]
        meta = {"sample_quality": [5.3, 9.1, 2.7, 5.8]}
        split = "training"

        dq.log_data_samples(texts=texts, labels=labels, ids=ids, meta=meta, split=split)
    """

    __logger_name__ = "text_classification"
    logger_config = text_classification_logger_config

    def __init__(
        self,
        texts: List[str] = None,
        labels: List[str] = None,
        ids: List[int] = None,
        split: str = None,
        meta: MetasType = None,
        inference_name: str = None,
    ) -> None:
        """Create data logger.

        :param texts: The raw text inputs for model training. List[str]
        :param labels: the ground truth labels aligned to each text field.
        List[str]
        :param ids: Optional unique indexes for each record. If not provided, will
        default to the index of the record. Optional[List[Union[int,str]]]
        :param split: The split for training/test/validation
        """
        super().__init__(meta)
        # Need to compare to None because they may be np arrays which cannot be
        # evaluated with bool directly
        self.texts = texts if texts is not None else []
        self.labels = [str(i) for i in labels] if labels is not None else []
        self.ids = ids if ids is not None else []
        self.split = split
        self.inference_name = inference_name

    def log_data_samples(
        self,
        *,
        texts: List[str],
        ids: List[int],
        labels: Optional[List[str]] = None,
        split: Optional[Split] = None,
        inference_name: Optional[str] = None,
        meta: Optional[MetasType] = None,
        **kwargs: Any,  # For typing
    ) -> None:
        """Log input samples for text classification

        ex:
        .. code-block:: python

            dq.init("text_classification")
            all_labels = ["A", "B", "C"]
            dq.set_labels_for_run(labels = all_labels)

            texts: List[str] = [
                "Text sample 1",
                "Text sample 2",
                "Text sample 3",
                "Text sample 4"
            ]

            labels: List[str] = ["B", "C", "A", "A"]

            ids: List[int] = [0, 1, 2, 3]
            split = "training"

            dq.log_data_samples(texts=texts, labels=labels, ids=ids, split=split)

        :param texts: List[str] text samples
        :param ids: List[int | str] IDs for each text sample
        :param labels: List[str] labels for each text sample.
            Required if not in inference
        :param split: train/test/validation/inference. Can be set here or via
            dq.set_split
        :param inference_name: If logging inference data, a name for this inference
            data is required. Can be set here or via dq.set_split
        :param meta: Dict[str, List[str, int, float]]. Metadata for each text sample
            Format is the {"metadata_field_name": [metdata value per sample]}
        """
        self.validate_kwargs(kwargs)
        self.texts = texts
        self.ids = ids
        self.split = split
        self.labels = [str(i) for i in labels] if labels is not None else []
        self.inference_name = inference_name
        self.meta = meta or {}
        self.log()

    def log_data_sample(
        self,
        *,
        text: str,
        id: int,
        label: Optional[str] = None,
        split: Optional[Split] = None,
        inference_name: Optional[str] = None,
        meta: Optional[MetaType] = None,
        **kwargs: Any,
    ) -> None:
        """
        Log a single input sample for text classification

        :param text: str the text sample
        :param id: The sample ID
        :param label: str label for the sample. Required if not in inference
        :param split: train/test/validation/inference. Can be set here or via
            dq.set_split
        :param inference_name: If logging inference data, a name for this inference
            data is required. Can be set here or via dq.set_split
        :param meta: Dict[str, Union[str, int, float]]. Metadata for the text sample
            Format is the {"metadata_field_name": metadata_field_value}
        """
        self.validate_kwargs(kwargs)
        self.texts = [text]
        self.ids = [id]
        self.split = split
        self.labels = [label] if label else []
        self.inference_name = inference_name
        self.meta = {i: [meta[i]] for i in meta} if meta else {}
        self.log()

    def log_dataset(
        self,
        dataset: DataSet,
        *,
        batch_size: int = ITER_CHUNK_SIZE,
        text: Union[str, int] = "text",
        id: Union[str, int] = "id",
        label: Optional[Union[str, int]] = "label",
        split: Optional[Split] = None,
        inference_name: Optional[str] = None,
        meta: Optional[List[Union[str, int]]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Log a dataset of input samples for text classification

        :param dataset: The dataset to log. This can be an python iterable or
            Pandas/Vaex dataframe. If an iterable, it can be a list of elements that can
            be indexed into either via int index (tuple/list) or string/key index (dict)
        :param batch_size: Number of samples to log in a batch. Default 100,000
        :param text: The key/index of the text fields
        :param id: The key/index of the id fields
        :param label: The key/index of the label fields
        :param split: train/test/validation/inference. Can be set here or via
            dq.set_split
        :param inference_name: If logging inference data, a name for this inference
            data is required. Can be set here or via dq.set_split
        :param meta: List[str, int]: The keys/indexes of each metadata field.
            Consider a pandas dataframe, this would be the list of columns corresponding
            to each metadata field to log
        """
        self.validate_kwargs(kwargs)
        self.split = split
        self.inference_name = inference_name
        meta = meta or []
        column_map = {text: "text", id: "id"}
        label = None if split == Split.inference else label
        if label:
            column_map[label] = "label"
        if isinstance(dataset, pd.DataFrame):
            dataset = dataset.rename(columns=column_map)
            self._log_df(dataset, meta)
        elif isinstance(dataset, DataFrame):
            for chunk in range(0, len(dataset), batch_size):
                chunk_df = dataset[chunk : chunk + batch_size]
                chunk_df = rename_df(chunk_df, column_map)
                self._log_df(chunk_df, meta)
        elif self.is_hf_dataset(dataset):
            self._log_hf_dataset(
                dataset, batch_size, text, id, meta, label, split, inference_name
            )
        elif isinstance(dataset, Iterable):
            self._log_iterator(
                dataset, batch_size, text, id, meta, label, split, inference_name
            )
        else:
            raise GalileoException(
                f"Dataset must be one of pandas, vaex, HF, or Iterable, "
                f"but got {type(dataset)}"
            )

    def _log_hf_dataset(
        self,
        dataset: Any,
        batch_size: int,
        text: Union[str, int],
        id: Union[str, int],
        meta: List[Union[str, int]],
        label: Union[str, int] = None,
        split: Split = None,
        inference_name: str = None,
    ) -> None:
        """Helper function to log a huggingface dataset

        HuggingFace datasets can be sliced, returning a dict that is in the correct
        format to log directly.
        """

        parse_label = lambda x: x  # noqa: E731
        # If label is integer, convert to string #

        if isinstance(dataset[0].get(label), int):
            try:
                parse_label = lambda x: dataset.features[label].int2str(x)  # noqa: E731
            except Exception:
                # TODO: Simplify this logic with mapping the int label to string ticket
                raise GalileoException(
                    "Your dataset does not have label names. Please include them"
                )

        assert dataset[0].get(id) is not None, GalileoException(
            f"id ({id}) field must be present in dataset"
        )

        for i in range(0, len(dataset), batch_size):
            chunk = dataset[i : i + batch_size]
            data = dict(
                text=chunk[text],
                id=chunk[id],
                label=parse_label(chunk[label]) if label else None,
            )
            chunk_meta = {col: chunk[col] for col in meta}
            self._log_dict(data, chunk_meta, split, inference_name)

    def _log_iterator(
        self,
        dataset: Iterable,
        batch_size: int,
        text: Union[str, int],
        id: Union[str, int],
        meta: List[Union[str, int]],
        label: Union[str, int] = None,
        split: Split = None,
        inference_name: str = None,
    ) -> None:
        batches = defaultdict(list)
        metas = defaultdict(list)
        for chunk in dataset:
            batches["text"].append(self._convert_tensor_to_py(chunk[text]))
            batches["id"].append(self._convert_tensor_to_py(chunk[id]))
            if label:
                # Process separately because multi-label needs to override this
                # to handle labels as a list of lists
                batches = self._process_label(batches, chunk[label])
            for meta_col in meta:
                metas[meta_col].append(self._convert_tensor_to_py(chunk[meta_col]))

            if len(batches["text"]) >= batch_size:
                self._log_dict(batches, metas, split, inference_name)
                batches.clear()
                metas.clear()
        # in case there are any left
        if batches:
            self._log_dict(batches, metas, split, inference_name)

    def _process_label(self, batches: DefaultDict, label: Any) -> DefaultDict:
        """Process label for text-classification and multi-label accordingly"""
        batches["label"].append(self._convert_tensor_to_py(label))
        return batches

    def _log_dict(
        self, d: Dict, meta: Dict, split: Split = None, inference_name: str = None
    ) -> None:
        self.log_data_samples(
            texts=d["text"],
            labels=d["label"],
            ids=d["id"],
            split=split,
            inference_name=inference_name,
            meta=meta,
        )

    def _log_df(
        self, df: Union[pd.DataFrame, DataFrame], meta: List[Union[str, int]]
    ) -> None:
        """Helper to log a pandas or vex df"""
        self.texts = df["text"].tolist()
        self.ids = df["id"].tolist()
        # Inference case
        if "label" in df:
            self.labels = df["label"].tolist()
        for meta_col in meta:
            self.meta[str(meta_col)] = df[meta_col].tolist()
        self.log()

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
        text_len = len(self.texts)
        id_len = len(self.ids)

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
        """Validates that the labels logged match the labels set"""
        self.logger_config.observed_labels.update(self.labels)
        found_labels = self.logger_config.observed_labels
        if self.logger_config.labels:
            set_labels = set(self.logger_config.labels)
            assert set_labels.issuperset(found_labels), (
                f"Labels set to {set_labels} but found logged labels {found_labels}. "
                f"Logged labels must be the same as labels set during "
                f"dq.set_labels_for_run. Fix logged data or update labels."
            )

    def _get_input_df(self) -> DataFrame:
        inp = dict(
            id=self.ids,
            text=self.texts,
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
        cls, df: DataFrame, prob_only: bool, split: str
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

        assert cls.logger_config.observed_labels.issubset(cls.logger_config.labels), (
            f"Labels set to {cls.logger_config.labels} do not align with observed "
            f"logged labels of {cls.logger_config.observed_labels}. Set labels must "
            "contain all logged labels. Update your labels with "
            "`dq.set_labels_for_run` or fix input data."
        )
