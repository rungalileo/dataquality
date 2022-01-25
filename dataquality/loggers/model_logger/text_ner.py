import json
from collections import defaultdict
from enum import Enum, unique
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

import numpy as np
import vaex
from vaex.dataframe import DataFrame

from dataquality.core._config import config
from dataquality.exceptions import GalileoException
from dataquality.loggers import BaseGalileoLogger
from dataquality.loggers.data_logger import BaseGalileoDataLogger
from dataquality.loggers.logger_config.text_ner import text_ner_logger_config
from dataquality.loggers.model_logger.base_model_logger import BaseGalileoModelLogger
from dataquality.schemas import __data_schema_version__
from dataquality.utils.vaex import _save_hdf5_file, _try_concat_df


@unique
class GalileoModelLoggerAttributes(str, Enum):
    gold_emb = "gold_emb"
    pred_emb = "pred_emb"
    pred_spans = "pred_spans"
    probs = "probs"
    ids = "ids"
    # mixin restriction on str (due to "str".split(...))
    split = "split"  # type: ignore
    epoch = "epoch"
    dep_scores = "dep_scores"

    @staticmethod
    def get_valid() -> List[str]:
        return list(map(lambda x: x.value, GalileoModelLoggerAttributes))


class TextNERModelLogger(BaseGalileoModelLogger):
    """
    Class for logging model output data of Text NER models to Galileo.

    * Gold Embeddings: List[List[List[np.array]]]. The Embeddings of the true span
    tokens for the text_tokenized. This is per token of a span. For each gold (true)
    span in a text sample, there may be 1 or more tokens. Each token will have an
    embedding vector.
    NOTE: Max 5 spans per text sample. More than 5 will throw an error
    * Pred Embeddings: List[List[List[np.array]]]. The Embeddings of the PREDICTED span
    tokens for the text_tokenized. This is per token of a span. For each predicted
    span in a text sample, there may be 1 or more tokens. Each token will have an
    embedding vector.
    NOTE: The pred embeddings will match the probabilities but may not match
    the gold embeddings (the model may predict the wrong span)
    NOTE: Max 5 spans per text sample. More than 5 will throw an error
    * Probabilities: List[List[List[np.array]]]. The prediction probabilities of each
    spans' tokens. For each span in a sentence, there will be 1 or more tokens. The
    model will have a probability vector for each token.
    NOTE: The probabilities will match the pred embeddings but may not match
    the gold embeddings (the model may predict the wrong span)
    NOTE: Max 5 spans per text sample. More than 5 will throw an error
    * ids: Indexes of each input field: List[int]. These IDs must align with the input
    IDs for each sample input. This will be used to join them together for analysis
    by Galileo.
    * split: The model training/test/validation split for the samples being logged

    ex: (see the data input example in the DataLogger for NER
    `print(dataquality.get_data_logger().__doc__))`
    .. code-block:: python

        # Logged with `dataquality.log_input_data`
        text_inputs: List[str] = [
            "The president is Joe Biden",
            "Joe Biden addressed the United States on Monday"
        ]
        TODO: Change format
        probs = [
            [
                [prob(joe), prob(bi), prob(##den)]  # Correct span
            ],
            [
                [prob(joe), prob(bi), prob(##den)],  # Correct span
                [prob(monday)]  # Incorrect span, but the prediction
            ]
        ]
        TODO: Change format
        pred_emb = [
            [
                [emb(joe), emb(bi), emb(##den)]  # Correct span
            ],
            [
                [emb(joe), emb(bi), emb(##den)],  # Correct span
                [emb(monday)]  # Incorrect span, but the prediction
            ]
        ]
        TODO: Change format
        gold_emb = [
            [
                [emb(joe), emb(bi), emb(##den)]  # True span
            ]
            [
                [emb(joe), emb(bi), emb(##den)],              # True span
                [emb(unite), emb(##d), emb(state), emb(##s)]  # True span
            ]
        ]
        epoch = 0
        ids = [0, 1]  # Must match the data input IDs
        split = "training"
        dataquality.log_model_outputs(
            gold_emb, pred_emb, pred_spans, probs, ids, split, epoch
        )
    """

    __logger_name__ = "text_ner"
    logger_config = text_ner_logger_config

    def __init__(
        self,
        # gold_emb: List[List[np.ndarray]] = None,
        # pred_emb: List[List[np.ndarray]] = None,
        emb: List[np.ndarray] = None,
        # pred_spans: List[List[dict]] = None,
        probs: List[np.ndarray] = None,
        ids: Union[List, np.ndarray] = None,
        split: Optional[str] = None,
        epoch: Optional[int] = None,
    ) -> None:
        super().__init__()
        # Need to compare to None because they may be np arrays which cannot be
        # evaluated with bool directly
        # self.gold_emb = gold_emb if gold_emb is not None else []
        # self.pred_emb = pred_emb if pred_emb is not None else []
        self.emb = emb if emb is not None else []
        # self.pred_spans = pred_spans if pred_spans is not None else []
        self.probs = probs if probs is not None else []
        self.ids = ids if ids is not None else []
        self.split = split
        self.epoch = epoch

        # Calculated internally
        self.gold_emb: List[List[np.ndarray]] = []
        self.pred_emb: List[List[np.ndarray]] = []
        self.pred_spans: List[List[Dict]] = []
        self.dep_scores_gold: List[List[float]] = []
        self.dep_scores_pred: List[List[float]] = []

    @staticmethod
    def get_valid_attributes() -> List[str]:
        """
        Returns a list of valid attributes that GalileoModelConfig accepts
        :return: List[str]
        """
        return GalileoModelLoggerAttributes.get_valid()

    def validate(self) -> None:
        """
        Validates that the current config is correct.
        * emb, probs, and ids must exist and be the same length
        :return:
        """
        super().validate()

        # Add the input data dataframe to the logger config so we can use it to eagerly
        # calculate DEP score
        if not self.logger_config.input_data:
            df_dir = (
                f"{BaseGalileoLogger.LOG_FILE_DIR}/{config.current_project_id}/"
                f"{config.current_run_id}/{BaseGalileoDataLogger.INPUT_DATA_NAME}"
            )
            self.logger_config.input_data = vaex.open(df_dir).copy()

        emb_len = len(self.emb)
        prob_len = len(self.probs)
        id_len = len(self.ids)

        # self.emb = self._convert_tensor_ndarray(self.emb, "Embedding")
        # self.probs = self._convert_tensor_ndarray(self.probs, "Prob")
        # self.ids = self._convert_tensor_ndarray(self.ids)

        assert all([emb_len, prob_len, id_len]), (
            f"All of emb, probs, and ids for your logger must be set, but "
            f"got emb:{bool(emb_len)}, probs:{bool(prob_len)}, ids:{bool(id_len)}"
        )

        assert emb_len == prob_len == id_len, (
            f"All of emb, probs, and ids for your logger must be the same "
            f"length, but got (emb, probs, ids) -> ({emb_len}, {prob_len}, {id_len})"
        )

        # We need to average the embeddings for the tokens within a span
        # so each span has only 1 embedding vector
        avg_gold_emb: List[List[np.ndarray]] = []
        avg_pred_emb: List[List[np.ndarray]] = []
        dep_scores: List[List[float]] = []
        for sample_id, sample_emb, sample_prob in zip(self.ids, self.emb, self.probs):
            # err = (
            #     f"Cannot have more than {self.logger_config.max_spans} spans in a "
            #     "sample, but had {}"
            # )
            # assert len(span_emb) < self.logger_config.max_spans, err.format(
            #     len(span_emb)
            # )
            # assert len(pred_span_probs) < self.logger_config.max_spans, err.format(
            #     len(pred_span_probs)
            # )
            # avg_gold_emb.append(
            #     [np.mean(gold_span, axis=0) for gold_span in gold_span_emb]
            # )
            # avg_pred_emb.append(
            #     [np.mean(pred_span, axis=0) for pred_span in pred_span_emb]
            # )

            # TODO: Eagerly calculate the DEP score and discard the probs
            sample_pred_spans = self._extract_pred_spans(sample_prob)
            self.pred_spans.append(sample_pred_spans)
            pred_dep = self._calculate_dep_scores(sample_prob, sample_pred_spans)
            self.dep_scores_pred.append(pred_dep)
            # TODO: Ben - get the gold span
            gold_span_tup = self.logger_config.gold_spans[sample_id]
            sample_gold_spans: List[Dict] = [
                dict(start=start, end=end, label=label)
                for start, end, label in gold_span_tup
            ]
            gold_dep = self._calculate_dep_scores(sample_prob, sample_gold_spans)
            self.dep_scores_gold.append(gold_dep)

            # TODO: Get gold and pred embeddings
            gold_emb = self._extract_embeddings(sample_gold_spans, sample_emb)
            self.gold_emb.append(gold_emb)
            pred_emb = self._extract_embeddings(sample_pred_spans, sample_emb)
            self.pred_emb.append(pred_emb)

            if not self.logger_config.observed_num_labels:
                # TODO: get the observed num labels
                pass

        # Get the embedding shape. Filter out nulls
        if not self.logger_config.num_emb:
            emb = next(filter(lambda emb: not np.isnan(emb[0]).all(), self.gold_emb))[0]
            self.logger_config.num_emb = emb.shape[0]

    def _extract_embeddings(
        self, spans: List[Dict], emb: np.ndarray
    ) -> List[np.ndarray]:
        """Get the embeddings for each span, on a per-sample basis"""

    def _extract_pred_spans(self, pred_prob: np.ndarray) -> List[Dict]:
        pred_spans: List[Dict]
        self.logger_config.tagging_schema
        self.logger_config.labels
        # TODO: Nidhi extract pred spans

    def _calculate_dep_scores(
        self, pred_prob: np.ndarray, spans: List[Dict]
    ) -> List[float]:
        """Calculates dep scores for each span on a per-sample basis"""
        self.logger_config.tagging_schema
        self.logger_config.labels

    def write_model_output(self, model_output: DataFrame) -> None:
        location = (
            f"{self.LOG_FILE_DIR}/{config.current_project_id}"
            f"/{config.current_run_id}"
        )

        epoch, split = model_output[["epoch", "split"]][0]
        path = f"{location}/{split}/{epoch}"
        object_name = f"{str(uuid4()).replace('-', '')[:12]}.hdf5"
        _save_hdf5_file(path, object_name, model_output)
        _try_concat_df(path)

    def _log(self) -> None:
        """Threaded logger target implemented by child"""
        try:
            self.validate()
        except AssertionError as e:
            raise GalileoException(f"The provided GalileoModelConfig is invalid. {e}")
        data = self._get_data_dict()
        self.write_model_output(model_output=vaex.from_dict(data))

    def _get_data_dict(self) -> Dict[str, Any]:
        data = defaultdict(list)
        for record_id, span_deps, gold_emb, pred_emb, pred_span in zip(
            self.ids, self.dep_scores, self.gold_emb, self.pred_emb, self.pred_spans
        ):
            record = {
                "id": record_id,
                "epoch": self.epoch,
                "split": self.split,
                "pred_span": json.dumps(pred_span),
                "num_pred_spans": len(pred_span),
                "num_gold_spans": len(gold_emb),
                "data_schema_version": __data_schema_version__,
            }
            for i, gold_span_emb in enumerate(gold_emb):
                record[f"gold_emb_{i}"] = gold_span_emb
            for i, pred_span_emb in enumerate(pred_emb):
                record[f"pred_emb_{i}"] = pred_span_emb
            for i, span_dep in enumerate(span_deps):
                record[f"dep_{i}"] = span_dep

            # Pad the embeddings and deps for missing values
            # We enable up to 5 spans, so for samples with less, we need to add values
            # those vaex df columns. We add np.zeros for embs and -1 for dep scores
            rng = range(record["num_pred_spans"], self.logger_config.max_spans)
            for i in rng:
                record[f"pred_emb_{i}"] = np.zeros(self.logger_config.num_emb)
                record[f"dep_{i}"] = -1

            rng = range(record["num_gold_spans"], self.logger_config.max_spans)
            for i in rng:
                record[f"gold_emb_{i}"] = np.zeros(self.logger_config.num_emb)

            for k in record.keys():
                data[k].append(record[k])
        return data

    def __setattr__(self, key: Any, value: Any) -> None:
        if key not in self.get_valid_attributes():
            raise AttributeError(
                f"{key} is not a valid attribute of {self.__logger_name__} logger. "
                f"Only {self.get_valid_attributes()}"
            )
        super().__setattr__(key, value)
