from collections import defaultdict
from enum import Enum, unique
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy import ndarray

from dataquality.loggers.logger_config.text_ner import (
    TaggingSchema,
    text_ner_logger_config,
)
from dataquality.loggers.model_logger.base_model_logger import BaseGalileoModelLogger
from dataquality.schemas import __data_schema_version__
from dataquality.schemas.ner_errors import NERErrorType


@unique
class GalileoModelLoggerAttributes(str, Enum):
    gold_emb = "gold_emb"
    gold_spans = "gold_spans"
    gold_dep = "gold_dep"
    emb = "emb"
    pred_emb = "pred_emb"
    pred_spans = "pred_spans"
    pred_dep = "pred_dep"
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

    * emb: List[np.ndarray]. TODO: Nidhi updated definition
    The Embeddings of the true span
    tokens for the text_tokenized. This is per token of a span. For each gold (true)
    span in a text sample, there may be 1 or more tokens. Each token will have an
    embedding vector.
    * Probabilities: List[np.ndarray]. TODO: Nidhi updated definition
    The prediction probabilities of each
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
    `dataquality.get_data_logger().doc()`
    .. code-block:: python

        # Logged with `dataquality.log_input_data`
        text_inputs: List[str] = [
            "The president is Joe Biden",
            "Joe Biden addressed the United States on Monday"
        ]
        TODO: Nidhi change format for prob
        probs = [
            [
                [prob(joe), prob(bi), prob(##den)]  # Correct span
            ],
            [
                [prob(joe), prob(bi), prob(##den)],  # Correct span
                [prob(monday)]  # Incorrect span, but the prediction
            ]
        ]
        TODO: Nidhi example for emb
        emb = [

        ]
        epoch = 0
        ids = [0, 1]  # Must match the data input IDs
        split = "training"
        dataquality.log_model_outputs(
            emb=emb, probs=probs, ids=ids, split=split, epoch=epoch
        )
    """

    __logger_name__ = "text_ner"
    logger_config = text_ner_logger_config

    def __init__(
        self,
        emb: List[np.ndarray] = None,
        probs: List[np.ndarray] = None,
        ids: Union[List, np.ndarray] = None,
        split: Optional[str] = None,
        epoch: Optional[int] = None,
    ) -> None:
        super().__init__()
        # Need to compare to None because they may be np arrays which cannot be
        # evaluated with bool directly
        self.emb = emb if emb is not None else []
        # self.pred_spans = pred_spans if pred_spans is not None else []
        self.probs = probs if probs is not None else []
        self.ids = ids if ids is not None else []
        self.split = split
        self.epoch = epoch

        # Calculated internally
        self.gold_emb: List[List[np.ndarray]] = []
        self.gold_spans: List[List[Dict]] = []
        self.gold_dep: List[List[float]] = []

        self.pred_emb: List[List[np.ndarray]] = []
        self.pred_spans: List[List[Dict]] = []
        self.pred_dep: List[List[float]] = []

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

        emb_len = len(self.emb)
        prob_len = len(self.probs)
        id_len = len(self.ids)

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
        logged_sample_ids = []
        for sample_id, sample_emb, sample_prob in zip(self.ids, self.emb, self.probs):
            # To extract metadata about the sample we are looking at
            span_key = self.logger_config.get_sample_key(str(self.split), sample_id)

            # TODO: Nidhi, you can pass this into your functions
            sample_token_len = self.logger_config.sample_length[span_key]

            # Get prediction spans
            sample_pred_spans = self._extract_pred_spans(sample_prob)
            print(f"Sample {sample_id} has {len(sample_pred_spans)} pred spans")
            # Get gold (ground truth) spans
            gold_span_tup = self.logger_config.gold_spans.get(span_key, [])
            sample_gold_spans: List[Dict] = [
                dict(start=start, end=end, label=label)
                for start, end, label in gold_span_tup
            ]
            # If there were no golds and no preds for a sample, don't log this output
            if not sample_pred_spans and not sample_gold_spans:
                continue

            gold_dep, pred_dep = self._calculate_dep_scores(
                sample_prob, sample_gold_spans, sample_pred_spans
            )
            gold_emb = self._extract_embeddings(sample_gold_spans, sample_emb)
            pred_emb = self._extract_embeddings(sample_pred_spans, sample_emb)

            logged_sample_ids.append(sample_id)
            self.pred_spans.append(sample_pred_spans)
            self.gold_spans.append(sample_gold_spans)
            self.pred_dep.append(pred_dep)
            self.gold_dep.append(gold_dep)
            self.gold_emb.append(gold_emb)
            self.pred_emb.append(pred_emb)

            if not self.logger_config.observed_num_labels:
                # TODO: Nidhi - is it possible to get the observed num labels from
                #  the input data? It doesnt need to be here, anywhere in the code
                pass

        self.ids = logged_sample_ids
        # Get the embedding shape. Filter out nulls
        if not self.logger_config.num_emb:
            emb = next(filter(lambda emb: not np.isnan(emb[0]).all(), self.gold_emb))[0]
            self.logger_config.num_emb = emb.shape[0]

    def _extract_embeddings(
        self, spans: List[Dict], emb: np.ndarray
    ) -> List[np.ndarray]:
        """Get the embeddings for each span, on a per-sample basis

        We take the average of the token embeddings per span and use that as the span
        level embedding
        """
        embeddings = []
        for span in spans:
            start = span["start"]
            end = span["end"]
            span_embeddings = emb[start:end, :]
            avg_span_embedding = span_embeddings.mean(axis=0)
            embeddings.append(avg_span_embedding)
        return embeddings

    def _extract_pred_spans(self, pred_prob: np.ndarray) -> List[Dict]:
        """
        TODO: Nidhi add description of logic
        """
        argmax_indices: List[int] = pred_prob.argmax(axis=1)
        pred_sequence: List[str] = [
            self.logger_config.labels[x] for x in argmax_indices
        ]
        if self.logger_config.tagging_schema == TaggingSchema.BIO:
            pred_spans = self._extract_pred_spans_bio(pred_sequence)
        elif self.logger_config.tagging_schema == TaggingSchema.BILOU:
            pred_spans = self._extract_pred_spans_bilou(pred_sequence)
        else:  # BIOES
            pred_spans = self._extract_pred_spans_bioes(pred_sequence)
        return pred_spans

    def _extract_pred_spans_bio(self, pred_sequence: List[str]) -> List[Dict]:
        """
        Converts the prediction sequences into prediction span tokens

        Final format is of {'start': int, 'end': int, 'label': str}

        The function looks for a token with a B tag, and then continues forward
        in the sequence looking for sequential I tags with the same label.

        example:

        pred_sequence = [
            'I-PER', 'B-PER', 'B-LOC', 'B-PER', 'B-PER', 'B-PER', 'B-PER', 'B-PER',
            'B-PER', 'I-PER', 'B-PER', 'B-PER', 'B-PER', 'B-PER', 'B-PER', 'B-PER',
            'I-LOC', 'I-PER', 'I-LOC', 'I-PER', 'B-PER', 'B-PER', 'B-PER', 'B-PER',
            'B-ORG', 'O', 'B-PER', 'I-LOC', 'B-PER', 'I-ORG'
        ]
        self._extract_pred_spans_bio(pred_sequence)
        >> [
            {'start': 1, 'end': 2, 'label': 'PER'},
            {'start': 2, 'end': 3, 'label': 'LOC'},
            {'start': 3, 'end': 4, 'label': 'PER'},
            {'start': 4, 'end': 5, 'label': 'PER'},
            {'start': 5, 'end': 6, 'label': 'PER'},
            {'start': 6, 'end': 7, 'label': 'PER'},
            {'start': 7, 'end': 8, 'label': 'PER'},
            {'start': 8, 'end': 10, 'label': 'PER'},
            {'start': 10, 'end': 11, 'label': 'PER'},
        ]
        """
        pred_spans = []
        total_b_count = 0
        idx = 0
        # Use a while loop so we can skip rows already scanned in the inner loop
        while idx < len(pred_sequence):
            token = pred_sequence[idx]
            next_idx = idx + 1

            if not token.startswith("B"):
                idx += 1
                continue

            # We've hit a prediction. Continue until it's invalid
            # Invalid means we hit a new B or O tag, or the next I tag has a
            # different label
            total_b_count += 1
            token_val, token_label = token.split("-")

            for next_tok in pred_sequence[idx + 1 :]:
                # next_tok == "I" and the label matches the current B label
                if next_tok.startswith("I") and next_tok.split("-")[1] == token_label:
                    next_idx += 1
                else:
                    break

            pred_spans.append({"start": idx, "end": next_idx, "label": token_label})
            idx = next_idx

        assert total_b_count == len(pred_spans)
        return pred_spans

    def _extract_pred_spans_bilou(self, pred_sequence: List[str]) -> List[Dict]:
        """
        convert BILOU into BIO and extract the spans

        L becomes I
        U becomes B
        """
        for idx, label in enumerate(pred_sequence):
            if pred_sequence[idx].split("-")[0] == "L":
                pred_sequence[idx] = pred_sequence[idx].replace("L", "I", 1)
            elif pred_sequence[idx].split("-")[0] == "U":
                pred_sequence[idx] = pred_sequence[idx].replace("U", "B", 1)
        return self._extract_pred_spans_bio(pred_sequence)

    def _extract_pred_spans_bioes(self, pred_sequence: List[str]) -> List[Dict]:
        """convert BIOES into BIO and extract the spans

        E becomes I
        S becomes B
        """
        for idx, label in enumerate(pred_sequence):
            if pred_sequence[idx].split("-")[0] == "E":
                pred_sequence[idx] = pred_sequence[idx].replace("E", "I", 1)
            elif pred_sequence[idx].split("-")[0] == "S":
                pred_sequence[idx] = pred_sequence[idx].replace("S", "B", 1)
        return self._extract_pred_spans_bio(pred_sequence)

    def _construct_gold_sequence(
        self, len_sequence: int, gold_spans: List[Dict]
    ) -> List[str]:
        """
        Using gold spans and tagging schema, construct the underlying gold sequence
        e.g. gold_spans = [{start=5, end=8, label="nothing"}]
        gold_sequence = [O, O, O, O, O, B-nothing, I-nothing, I-nothing, O, O] for BIO
        gold_sequence = [O, O, O, O, O, B-nothing, I-nothing, L-nothing, O, O] for BILOU
        """
        gold_sequence = ["O"] * len_sequence
        for span in gold_spans:
            start = span["start"]
            end = span["end"]
            label = span["label"]
            if self.logger_config.tagging_schema == TaggingSchema.BIO:
                gold_sequence[start:end] = [f"I-{label}"] * (end - start)
                gold_sequence[start] = f"B-{label}"
            elif self.logger_config.tagging_schema == TaggingSchema.BILOU:
                if end - start == 1:
                    gold_sequence[start] = f"U-{label}"
                else:
                    gold_sequence[start:end] = [f"I-{label}"] * (end - start)
                    gold_sequence[start] = f"B-{label}"
                    gold_sequence[end - 1] = f"L-{label}"
            elif self.logger_config.tagging_schema == TaggingSchema.BIOES:
                if end - start == 1:
                    gold_sequence[start] = f"S-{label}"
                else:
                    gold_sequence[start:end] = [f"I-{label}"] * (end - start)
                    gold_sequence[start] = f"B-{label}"
                    gold_sequence[end - 1] = f"E-{label}"
        return gold_sequence

    def _calculate_dep_score_across_spans(
        self, spans: List[Dict], dep_scores: List[float]
    ) -> List[float]:
        """TODO: Nidhi add description of logic"""
        dep_score_per_span = []
        for span in spans:
            start = span["start"]
            end = span["end"]
            dep_score_per_span.append(max(dep_scores[start:end]))
        return dep_score_per_span

    def _calculate_dep_scores(
        self, pred_prob: np.ndarray, gold_spans: List[Dict], pred_spans: List[Dict]
    ) -> Tuple[List[float], List[float]]:
        """Calculates dep scores for each span on a per-sample basis

        TODO: Nidhi add description of logic
        """
        label2idx = {l: i for i, l in enumerate(self.logger_config.labels)}
        argmax_indices: List[int] = pred_prob.argmax(axis=1).tolist()
        pred_sequence: List[str] = [
            self.logger_config.labels[x] for x in argmax_indices
        ]
        gold_sequence = self._construct_gold_sequence(len(pred_sequence), gold_spans)

        # Compute dep scores of all tokens in the sentence
        dep_scores_tokens = []
        for idx, token in enumerate(gold_sequence):
            g_label_idx = label2idx[token]  # index of ground truth
            token_prob_vector = pred_prob[idx]
            ordered_prob_vector = token_prob_vector.argsort()
            # We want the index of the highest probability that IS NOT the true label
            if ordered_prob_vector[-1] == g_label_idx:
                # Take the second highest probability because the highest was the label
                second_idx = ordered_prob_vector[-2]
            else:
                second_idx = ordered_prob_vector[-1]
            aum = token_prob_vector[g_label_idx] - token_prob_vector[second_idx]
            dep = (1 - aum) / 2  # normalize aum to dep
            dep_scores_tokens.append(dep)

        # Compute dep scores of all spans (max of dep score of tokens in a span)
        gold_dep = self._calculate_dep_score_across_spans(gold_spans, dep_scores_tokens)
        pred_dep = self._calculate_dep_score_across_spans(pred_spans, dep_scores_tokens)
        return gold_dep, pred_dep

    def _get_data_dict(self) -> Dict[str, Any]:
        """Format row data for storage with vaex/hdf5

        In NER, rows are stored at the span level, not the sentence level, so we
        will have repeating "sentence_id" values, which is OK. We will also loop
        through the data twice, once for prediction span information
        (one of pred_span, pred_emb, pred_dep per span) and once for gold span
        information (one of gold_span, gold_emb, gold_dep per span)
        """

        def _construct_span_row(
                d: defaultdict, id: int, start: int, end: int, dep: float, emb: ndarray
        ) -> defaultdict:
            d["sample_id"].append(id)
            d["epoch"].append(self.epoch)
            d["split"].append(self.split)
            d["data_schema_version"].append(__data_schema_version__)
            d["span_start"].append(start)
            d["span_end"].append(end)
            d["data_error_potential"].append(dep)
            d["emb"].append(emb)
            return d

        data: defaultdict = defaultdict(list)

        # Loop through samples
        num_samples = len(self.ids)
        for idx in range(num_samples):
            sample_id = self.ids[idx]
            gold_spans = self.gold_spans[idx]
            gold_embs = self.gold_emb[idx]
            gold_deps = self.gold_dep[idx]
            pred_spans = self.pred_spans[idx]
            pred_embs = self.pred_emb[idx]
            pred_deps = self.pred_dep[idx]

            # We want to dedup gold and prediction spans, as many will match on
            # index. When the index matches, the embeddings and dep score will too,
            # so we only log the span once and flag it as both gold and pred
            pred_span_inds = [(i["start"], i["end"]) for i in pred_spans]
            pred_spans_check = set(pred_span_inds)
            # Loop through the gold spans
            for gold_span, gold_emb, gold_dep in zip(gold_spans, gold_embs, gold_deps):
                span_ind = (gold_span["start"], gold_span["end"])
                data = _construct_span_row(
                    data, sample_id, span_ind[0], span_ind[1], gold_dep, gold_emb
                )
                data["is_gold"].append(True)
                data["gold"].append(gold_span["label"])

                if span_ind in pred_spans_check:
                    # Remove element from preds so it's not logged twice
                    ind = pred_span_inds.index(span_ind)
                    ps = pred_spans.pop(ind)
                    pred_embs.pop(ind)
                    pred_deps.pop(ind)
                    pred_span_inds.pop(ind)

                    data["is_pred"].append(True)
                    data["pred"].append(ps["label"])

                    # If indices match and tag doesn't, error_type is wrong_tag
                    error_type = (
                        NERErrorType.wrong_tag.value
                        if gold_span["label"] != ps["label"]
                        else NERErrorType.none.value
                    )
                    data["galileo_error_type"].append(error_type)

                else:
                    data["is_pred"].append(False)
                    data["pred"].append(None)
                    error_type = self._get_span_error_type(
                        pred_span_inds, span_ind
                    ).value
                    data["galileo_error_type"].append(error_type)

            # Loop through the remaining pred spans
            for pred_span, pred_emb, pred_dep in zip(pred_spans, pred_embs, pred_deps):
                start, end = pred_span["start"], pred_span["end"]
                data = _construct_span_row(
                    data, sample_id, start, end, pred_dep, pred_emb
                )
                data["is_gold"].append(False)
                data["is_pred"].append(True)
                data["pred"].append(pred_span["label"])
                data["gold"].append(None)
                # Pred only spans don't have an error type
                data["galileo_error_type"].append(NERErrorType.none.value)

        return data

    def _get_span_error_type(
        self, pred_spans: List[Tuple[int, int]], gold_span: Tuple[int, int]
    ) -> NERErrorType:
        """Determines the proper span error

        When indices don't match, the error is either span_shift or missed_label
        * span_shift: overlapping span start/end indices
        * missed_label: no overlap between span start/end indices

        In this function, we only look at spans that don't have pred/gold alignment,
        so the error can only be span_shift or missed_label.
        """
        gold_start, gold_end = gold_span
        # We start by assuming missed_label (because it's the worst case)
        # and update if we see overlap
        error_type = NERErrorType.missed_label
        for pred_span in pred_spans:
            pred_start, pred_end = pred_span
            if (
                gold_start <= pred_start <= gold_end
                or pred_start <= gold_start <= pred_end
            ):
                error_type = NERErrorType.span_shift
                break
        return error_type

    def __setattr__(self, key: Any, value: Any) -> None:
        if key not in self.get_valid_attributes():
            raise AttributeError(
                f"{key} is not a valid attribute of {self.__logger_name__} logger. "
                f"Only {self.get_valid_attributes()}"
            )
        super().__setattr__(key, value)
