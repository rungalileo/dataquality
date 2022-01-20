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
    * Pred Embeddings: List[List[List[np.array]]]. The Embeddings of the PREDICTED span
    tokens for the text_tokenized. This is per token of a span. For each predicted
    span in a text sample, there may be 1 or more tokens. Each token will have an
    embedding vector.
    NOTE: The pred embeddings will match the probabilities but may not match
    the gold embeddings (the model may predict the wrong span)
    * Probabilities: List[List[List[np.array]]]. The prediction probabilities of each
    spans' tokens. For each span in a sentence, there will be 1 or more tokens. The
    model will have a probability vector for each token.
    NOTE: The probabilities will match the pred embeddings but may not match
    the gold embeddings (the model may predict the wrong span)
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
        probs = [
            [
                [prob(joe), prob(bi), prob(##den)]  # Correct span
            ],
            [
                [prob(joe), prob(bi), prob(##den)],  # Correct span
                [prob(monday)]  # Incorrect span, but the prediction
            ]
        ]
        pred_spans = [
            [
                {"start":4, "end":7, "label":"person"}  # [joe], [bi, ##den]
            ],
            [
                {"start":0, "end":3, "label":"person"},    # [joe], [bi, ##den]
                {"start":11, "end":12, "label":"person"}  # [monday] (bad prediction)
            ]
        ]
        pred_emb = [
            [
                [emb(joe), emb(bi), emb(##den)]  # Correct span
            ],
            [
                [emb(joe), emb(bi), emb(##den)],  # Correct span
                [emb(monday)]  # Incorrect span, but the prediction
            ]
        ]
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
        gold_emb: List[List[np.ndarray]] = None,
        pred_emb: List[List[np.ndarray]] = None,
        pred_spans: List[List[List[np.ndarray]]] = None,
        probs: Union[List, np.ndarray] = None,
        ids: Union[List, np.ndarray] = None,
        split: Optional[str] = None,
        epoch: Optional[int] = None,
    ) -> None:
        super().__init__()
        # Need to compare to None because they may be np arrays which cannot be
        # evaluated with bool directly
        self.gold_emb = gold_emb if gold_emb is not None else []
        self.pred_emb = pred_emb if pred_emb is not None else []
        self.pred_spans = pred_spans if pred_spans is not None else []
        self.probs = probs if probs is not None else []
        self.ids = ids if ids is not None else []
        self.dep_scores: List[List[int]] = []
        self.split = split
        self.epoch = epoch

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

        gold_emb_len = len(self.gold_emb)
        pred_emb_len = len(self.pred_emb)
        pred_span_len = len(self.pred_spans)
        prob_len = len(self.probs)
        id_len = len(self.ids)

        # self.emb = self._convert_tensor_ndarray(self.emb, "Embedding")
        # self.probs = self._convert_tensor_ndarray(self.probs, "Prob")
        # self.ids = self._convert_tensor_ndarray(self.ids)

        assert all([gold_emb_len, pred_emb_len, pred_span_len, prob_len, id_len]), (
            f"All of emb, probs, and ids for your GalileoModelConfig must be set, but "
            f"got gold_emb:{bool(gold_emb_len)}, pred_emb:{bool(pred_emb_len)}, "
            f"pred_spans:{bool(pred_span_len)}  probs:{bool(prob_len)}, "
            f"ids:{bool(id_len)}"
        )

        assert gold_emb_len == pred_emb_len == pred_span_len == prob_len == id_len, (
            f"All of emb, probs, and ids for your GalileoModelConfig must be the same "
            f"length, but got (gold_emb, pred_emb, pred_span, probs, ids) -> "
            f"({gold_emb_len}, {pred_emb_len}, {pred_span_len}, {prob_len}, {id_len})"
        )

        # We need to average the embeddings for the tokens within a span
        # so each span has only 1 embedding vector
        avg_gold_emb: List[List[np.ndarray]] = []
        avg_pred_emb: List[List[np.ndarray]] = []
        dep_scores: List[List[int]] = []
        for gold_spans, pred_spans, prob_spans in zip(
            self.gold_emb, self.pred_emb, self.probs
        ):
            err = (
                f"Cannot have more than {self.logger_config.max_spans} spans in a "
                "sample, but had {}"
            )
            assert len(gold_spans) < self.logger_config.max_spans, err.format(
                len(gold_spans)
            )
            assert len(pred_spans) < self.logger_config.max_spans, err.format(
                len(pred_spans)
            )
            assert len(prob_spans) < self.logger_config.max_spans, err.format(
                len(prob_spans)
            )
            avg_gold_emb.append(
                [np.mean(gold_span, axis=0) for gold_span in gold_spans]
            )
            avg_pred_emb.append(
                [np.mean(pred_span, axis=0) for pred_span in pred_spans]
            )
            # TODO: Eagerly calculate the DEP score and discard the probs
        self.gold_emb = avg_gold_emb
        self.pred_emb = avg_pred_emb
        self.dep_scores = dep_scores

        # Get the embedding shape. Filter out nulls
        emb = next(filter(lambda emb: not np.isnan(emb[0]).all(), self.gold_emb))[0]
        self.logger_config.num_emb = emb.shape[0]

    def write_model_output(self, model_output: DataFrame) -> None:
        location = (
            f"{self.LOG_FILE_DIR}/{config.current_project_id}"
            f"/{config.current_run_id}"
        )

        self._set_num_labels(model_output)
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

    def _set_num_labels(self, df: DataFrame) -> None:
        self.logger_config.observed_num_labels = len(df[:1]["prob"].values[0])

    def __setattr__(self, key: Any, value: Any) -> None:
        if key not in self.get_valid_attributes():
            raise AttributeError(
                f"{key} is not a valid attribute of {self.__logger_name__} logger. "
                f"Only {self.get_valid_attributes()}"
            )
        super().__setattr__(key, value)
