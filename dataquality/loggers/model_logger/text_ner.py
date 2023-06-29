from collections import defaultdict
from enum import Enum, unique
from typing import Any, DefaultDict, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy import ndarray

from dataquality.exceptions import LogBatchError
from dataquality.loggers.logger_config.text_ner import text_ner_logger_config
from dataquality.loggers.model_logger.base_model_logger import BaseGalileoModelLogger
from dataquality.schemas import __data_schema_version__
from dataquality.schemas.ner import NERErrorType, NERProbMethod, TaggingSchema
from dataquality.schemas.split import Split
from dataquality.utils.ml import select_span_token_for_prob


@unique
class GalileoModelLoggerAttributes(str, Enum):
    gold_emb = "gold_emb"
    gold_spans = "gold_spans"
    gold_conf_prob = "gold_conf_prob"
    gold_loss_prob = "gold_loss_prob"
    gold_loss_prob_label = "gold_loss_prob_label"
    embs = "embs"
    pred_emb = "pred_emb"
    pred_spans = "pred_spans"
    pred_conf_prob = "pred_conf_prob"
    pred_loss_prob = "pred_loss_prob"
    pred_loss_prob_label = "pred_loss_prob_label"
    probs = "probs"
    logits = "logits"
    ids = "ids"
    # mixin restriction on str (due to "str".split(...))
    split = "split"  # type: ignore
    epoch = "epoch"
    log_helper_data = "log_helper_data"
    inference_name = "inference_name"

    @staticmethod
    def get_valid() -> List[str]:
        return list(map(lambda x: x.value, GalileoModelLoggerAttributes))


class TextNERModelLogger(BaseGalileoModelLogger):
    """
    Class for logging model output data of Text NER models to Galileo.

    * embs: List[np.ndarray]: Each np.ndarray represents all embeddings of a given
    sample. These embeddings are from the tokenized text, and will align with the tokens
    in the sample. If you have 12 samples in the dataset, with each sample of 20 tokens
    in length, and an embedding vector of size 768, len(embs) will be 12, and
    np.ndarray.shape is (20, 768).

    * logits: List[np.ndarray]: The NER prediction logits from the model
    for each token. These outputs are from the tokenized text, and will align with
    the tokens in the sample. If you have 12 samples in the dataset, with each sample
    of 20 tokens in length, and observed_num_labels as 40, len(probs) will be 12,
    and np.ndarray.shape is (20, 40).

    * probs: Probabilities: List[np.ndarray]: deprecated, use logits

    * ids: List[int]: These IDs must align with the input
    IDs for each sample input. This will be used to join them together for analysis
    by Galileo.

    * split: The model training/test/validation split for the samples being logged

    ex: (see the data input example in the DataLogger for NER
    `dataquality.get_data_logger().doc()`
    .. code-block:: python

        # Logged with `dataquality.log_model_outputs`
        logits =
            [np.array([model(the), model(president), model(is), model(joe),
            model(bi), model(##den), model(<pad>), model(<pad>), model(<pad>)]),
            np.array([model(joe), model(bi), model(##den), model(addressed),
            model(the), model(united), model(states), model(on), model(monday)])]

        embs =
            [np.array([emb(the), emb(president), emb(is), emb(joe),
            emb(bi), emb(##den), emb(<pad>), emb(<pad>), emb(<pad>)]),
            np.array([emb(joe), emb(bi), emb(##den), emb(addressed),
            emb(the), emb(united), emb(states), emb(on), emb(monday)])]

        epoch = 0
        ids = [0, 1]  # Must match the data input IDs
        split = "training"
        dataquality.log_model_outputs(
            embs=embs, logits=logits, ids=ids, split=split, epoch=epoch
        )
    """

    __logger_name__ = "text_ner"
    logger_config = text_ner_logger_config

    def __init__(
        self,
        embs: Optional[List[np.ndarray]] = None,
        probs: Optional[List[np.ndarray]] = None,
        logits: Optional[List[np.ndarray]] = None,
        ids: Optional[Union[List, np.ndarray]] = None,
        split: str = "",
        epoch: Optional[int] = None,
        inference_name: Optional[str] = None,
        labels: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__(
            embs=embs,
            probs=probs,
            logits=logits,
            ids=ids,
            split=split,
            epoch=epoch,
            inference_name=inference_name,
        )

        # Calculated internally
        self.gold_emb: List[List[np.ndarray]] = []
        self.gold_spans: List[List[Dict]] = []
        self.gold_conf_prob: List[List[np.ndarray]] = []
        self.gold_loss_prob: List[List[np.ndarray]] = []
        self.gold_loss_prob_label: List[List[int]] = []

        self.pred_emb: List[List[np.ndarray]] = []
        self.pred_spans: List[List[Dict]] = []
        self.pred_conf_prob: List[List[np.ndarray]] = []
        self.pred_loss_prob: List[List[np.ndarray]] = []
        self.pred_loss_prob_label: List[List[int]] = []
        # Used for helper data, does not get logged
        self.log_helper_data: Dict[str, Any] = {}

    @staticmethod
    def get_valid_attributes() -> List[str]:
        """
        Returns a list of valid attributes that GalileoModelConfig accepts
        :return: List[str]
        """
        return GalileoModelLoggerAttributes.get_valid()

    def validate_and_format(self) -> None:
        """
        Validates that the current config is correct.
        * embs, probs, and ids must exist and be the same length
        :return:
        """
        if len(self.logits):
            self.probs = self.convert_logits_to_probs(self.logits)
        elif len(self.probs):
            self.probs = self._convert_tensor_ndarray(self.probs)

        embs_len = len(self.embs)
        probs_len = len(self.probs)
        ids_len = len(self.ids)

        self.ids = self._convert_tensor_ndarray(self.ids)
        self.embs = self._convert_tensor_ndarray(self.embs)

        assert all([embs_len, probs_len, ids_len]), (
            f"All of emb, probs, and ids for your logger must be set, but "
            f"got emb:{bool(embs_len)}, probs:{bool(probs_len)}, ids:{bool(ids_len)}"
        )

        assert embs_len == probs_len == ids_len, (
            f"All of emb, probs, and ids for your logger must be the same "
            f"length, but got (emb, probs, ids) -> ({embs_len}, {probs_len}, {ids_len})"
        )

        # We need to average the embeddings for the tokens within a span
        # so each span has only 1 embedding vector
        logged_sample_ids = []
        for sample_id, sample_emb, sample_prob in zip(self.ids, self.embs, self.probs):
            # This will return True if there was a prediction or gold span
            if self._process_sample(sample_id, sample_emb, sample_prob):
                logged_sample_ids.append(sample_id)

        self.ids = logged_sample_ids
        if not self.ids:
            raise LogBatchError(
                "No samples in this batch had any gold or prediction spans. "
                "Logging will be skipped"
            )

    def _process_sample(
        self, sample_id: int, sample_emb: np.ndarray, sample_prob: np.ndarray
    ) -> bool:
        """Processes a sample. Returns whether or not the sample should be logged

        A sample should be logged only if there was at least 1 prediction span or 1
        gold span

        For inference mode, only prediction spans should be logged.
        """
        # To extract metadata about the sample we are looking at
        sample_key = self.logger_config.get_sample_key(Split(self.split), sample_id)

        # Unpadded length of the sample. Used to extract true predicted spans
        # which are padded by the model
        sample_token_len = self.logger_config.sample_length[sample_key]
        # Remove padding from prob vector as well
        sample_prob = sample_prob[:sample_token_len]
        # Get prediction spans
        sample_pred_spans = self._extract_pred_spans(sample_prob)
        # Get gold (ground truth) spans
        gold_span_tup = []
        if self.split != Split.inference:
            gold_span_tup = self.logger_config.gold_spans.get(sample_key, [])

        sample_gold_spans = [
            dict(start=start, end=end, label=label)
            for start, end, label in gold_span_tup
        ]

        # If there were no golds and no preds for a sample, don't log this sample
        if not sample_pred_spans and not sample_gold_spans:
            return False

        pred_emb = self._extract_span_embeddings(sample_pred_spans, sample_emb)
        pred_conf_prob, _ = self._extract_span_probs(
            sample_pred_spans, sample_prob, NERProbMethod.confidence
        )

        self.pred_spans.append(sample_pred_spans)
        self.pred_emb.append(pred_emb)
        self.pred_conf_prob.append(pred_conf_prob)

        if self.split != Split.inference:
            # If we are not in inference mode, we also have gold spans and
            # span probabilities
            gold_emb = self._extract_span_embeddings(sample_gold_spans, sample_emb)
            gold_sequence = self._construct_gold_sequence(
                len(sample_prob), sample_gold_spans
            )
            gold_conf_prob, _ = self._extract_span_probs(
                sample_gold_spans, sample_prob, NERProbMethod.confidence
            )
            gold_loss_prob, gold_loss_label = self._extract_span_probs(
                sample_gold_spans,
                sample_prob,
                NERProbMethod.loss,
                gold_sequence_str=gold_sequence,
            )
            pred_sequence_idx = sample_prob.argmax(axis=1)
            pred_loss_prob, pred_gold_label = self._extract_span_probs(
                sample_pred_spans,
                sample_prob,
                NERProbMethod.loss,
                gold_sequence_idx=pred_sequence_idx,
            )

            self.gold_spans.append(sample_gold_spans)
            self.gold_emb.append(gold_emb)
            self.gold_conf_prob.append(gold_conf_prob)
            self.gold_loss_prob.append(gold_loss_prob)
            self.pred_loss_prob.append(pred_loss_prob)
            self.gold_loss_prob_label.append(gold_loss_label)
            self.pred_loss_prob_label.append(pred_gold_label)

        return True

    def _extract_span_embeddings(
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

    def _extract_span_probs(
        self,
        spans: List[Dict],
        prob: np.ndarray,
        method: NERProbMethod,
        gold_sequence_str: Optional[List[str]] = None,
        gold_sequence_idx: Optional[np.ndarray] = None,
    ) -> Tuple[List[np.ndarray], List[int]]:
        """Get the probs for each span, on a per-sample basis

        Parameters
        ----------
        spans
            The spans to extract probs for
        prob
            The prob vector for the sample
        method
            The method to use to extract the probs. Can be either "confidence" or
            "loss"

        Returns
        -------
        List[np.ndarray]
            The probs for each span
        List[int]
            The gold label indices of token chosen for loss (needed for DEP calculation)
        """
        probs = []
        gold_labels = []
        if gold_sequence_idx is None and gold_sequence_str is not None:
            gold_sequence_idx = self.labels_to_idx(gold_sequence_str)

        has_gold_sequence = (
            gold_sequence_idx is not None and len(gold_sequence_idx) >= 0
        )

        for span in spans:
            start = span["start"]
            end = span["end"]
            span_probs = prob[start:end, :]
            # We ignore because if `has_gold_sequence` then has_gold_sequence must exist
            # (see above), but mypy can't figure that out
            span_gold_seq = (
                gold_sequence_idx[start:end]  # type: ignore
                if has_gold_sequence
                else None
            )
            # We select a token prob to represent the span prob
            span_prob, gold_label = select_span_token_for_prob(
                span_probs, method, span_gold_seq
            )
            probs.append(span_prob)
            # If method is 'loss' we return a list of gold labels
            # If method is 'confidence' we return an empty list which
            # will be ignored by the caller
            # We do this over returning a list of Nones for linting
            if gold_label is not None:
                gold_labels.append(gold_label)
        return probs, gold_labels

    def _extract_pred_spans(self, pred_prob: np.ndarray) -> List[Dict]:
        """
        Extract prediction labels from probabilities, and generate pred spans

        If the schema is non-BIO, we just first convert them into BIO then extract spans
        """
        # use length of the tokens stored to strip the pads
        # Drop the spans post first PAD
        argmax_indices: List[int] = pred_prob.argmax(axis=1)
        pred_sequence: List[str] = [
            self.logger_config.labels[x] for x in argmax_indices
        ]

        if self.logger_config.tagging_schema == TaggingSchema.BIO:
            pred_spans = self._extract_spans_bio(pred_sequence)
        else:  # BIOES or BILOU
            pred_spans = self._extract_spans_token_level(pred_sequence)
        return pred_spans

    def _extract_spans_bio(self, pred_sequence: List[str]) -> List[Dict]:
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
            token_val, token_label = self._split_token(token)

            for next_tok in pred_sequence[idx + 1 :]:
                # next_tok == "I" and the label matches the current B label
                if (
                    next_tok.startswith("I")
                    and self._split_token(next_tok)[1] == token_label
                ):
                    next_idx += 1
                else:
                    break

            pred_spans.append({"start": idx, "end": next_idx, "label": token_label})
            idx = next_idx

        assert total_b_count == len(pred_spans)
        return pred_spans

    def _extract_spans_token_level(self, sequence: List[str]) -> List[Dict]:
        """Extract spans at token or word level for gold or pred spans.

        This should be called with a BILOU or BIOES sequence
        """
        spans = []
        total_b_count = 0
        idx = 0
        found_end = False  # Tracks if there was an end tag predicted for BIOES

        # Use a while loop so we can skip rows already scanned in the inner loop
        while idx < len(sequence):
            token = sequence[idx]
            next_idx = idx + 1

            if self._is_single_token(token):
                total_b_count += 1
                # hit a single token prediction, update and continue
                token_val, token_label = self._split_token(token)
                spans.append({"start": idx, "end": next_idx, "label": token_label})
                idx += 1
                continue

            if not self._is_before_token(token):
                idx += 1
                continue

            # We've hit a prediction. Continue until it's invalid
            # Invalid means we hit a new B or O tag, or the next I tag has a
            # different label
            token_val, token_label = self._split_token(token)

            for next_tok in sequence[idx + 1 :]:
                # next_tok == "I" and the label matches the current B label
                if self._is_in_token(next_tok, token_label):
                    next_idx += 1
                # next_tok == "L"/"E" and the label matches the current B label
                elif self._is_end_token(next_tok, token_label):
                    next_idx += 1
                    found_end = True
                    total_b_count += 1
                    break
                else:
                    break
            if found_end:
                spans.append({"start": idx, "end": next_idx, "label": token_label})
            idx = next_idx
            found_end = False

        assert total_b_count == len(spans)
        return spans

    def _is_single_token(self, tok: str) -> bool:
        return tok.startswith("U") or tok.startswith("S")

    def _is_before_token(self, tok: str) -> bool:
        return tok.startswith("B")

    def _is_in_token(self, tok: str, label: str) -> bool:
        """We are inside a token if the token is an I tag and the label matches"""
        # next_tok == "I" and the label matches the current B label
        return tok.startswith("I") and self._split_token(tok)[1] == label

    def _is_end_token(self, tok: str, label: str) -> bool:
        """We are at the end of a token if L/E tag and the label matches"""
        return (tok.startswith("E") or tok.startswith("L")) and self._split_token(tok)[
            1
        ] == label

    def _split_token(self, tok: str) -> Tuple[str, str]:
        """Split the token value and label

        A token starts with a tag and has a label sepearated by "-"
        ie B-DOG or E-SOME_VAL or I-my-label
        but we only want to split on the first "-" incase the label itself has a "-"
        """
        tok_tag, tok_label = tok.split("-", maxsplit=1)
        return tok_tag, tok_label

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

    def _get_data_dict(self) -> Dict[str, Any]:
        """Format row data for storage with vaex/hdf5

        In NER, rows are stored at the span level, not the sentence level, so we
        will have repeating "sentence_id" values, which is OK. We will also loop
        through the data twice, once for prediction span information
        (one of pred_span, pred_emb, pred_prob per span) and once for gold span
        information (one of gold_span, gold_emb, gold_prob per span)

        NOTE: All spans are at the token level in this fn
        """
        if self.split == Split.inference:
            return self._get_data_dict_inference()

        data: defaultdict = defaultdict(list)

        # Loop through samples
        num_samples = len(self.ids)
        for idx in range(num_samples):
            sample_id = self.ids[idx]
            gold_spans = self.gold_spans[idx]
            gold_embs = self.gold_emb[idx]
            gold_conf_probs = self.gold_conf_prob[idx]
            gold_loss_probs = self.gold_loss_prob[idx]
            gold_loss_prob_labels = self.gold_loss_prob_label[idx]
            pred_spans = self.pred_spans[idx]
            pred_embs = self.pred_emb[idx]
            pred_conf_probs = self.pred_conf_prob[idx]
            pred_loss_probs = self.pred_loss_prob[idx]
            pred_loss_prob_labels = self.pred_loss_prob_label[idx]

            # We want to dedup gold and prediction spans, as many will match on
            # index. When the index matches, the embeddings and dep score will too,
            # so we only log the span once and flag it as both gold and pred
            pred_span_inds = [(i["start"], i["end"]) for i in pred_spans]
            pred_spans_check = set(pred_span_inds)
            # Loop through the gold spans
            for (
                gold_span,
                gold_emb,
                gold_conf_prob,
                gold_loss_prob,
                gold_loss_prob_label,
            ) in zip(
                gold_spans,
                gold_embs,
                gold_conf_probs,
                gold_loss_probs,
                gold_loss_prob_labels,
            ):
                data = self._construct_gold_span_row(
                    data,
                    sample_id,
                    gold_span,
                    gold_emb,
                    gold_conf_prob,
                    gold_loss_prob,
                    gold_loss_prob_label,
                )

                span_ind = (gold_span["start"], gold_span["end"])
                if span_ind in pred_spans_check:
                    # Remove element from preds so it's not logged twice
                    ind = pred_span_inds.index(span_ind)
                    ps = pred_spans.pop(ind)
                    pred_embs.pop(ind)
                    pred_conf_probs.pop(ind)
                    pred_loss_probs.pop(ind)
                    pred_loss_prob_labels.pop(ind)
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
                    data["pred"].append("")
                    error_type = self._get_span_error_type(
                        pred_span_inds, span_ind
                    ).value
                    data["galileo_error_type"].append(error_type)

            # Loop through the remaining pred spans
            for (
                pred_span,
                pred_emb,
                pred_conf_prob,
                pred_loss_prob,
                pred_loss_prob_label,
            ) in zip(
                pred_spans,
                pred_embs,
                pred_conf_probs,
                pred_loss_probs,
                pred_loss_prob_labels,
            ):
                data = self._construct_pred_span_row(
                    data,
                    sample_id,
                    pred_span,
                    pred_emb,
                    pred_conf_prob,
                    pred_loss_prob,
                    pred_loss_prob_label,
                    gold_spans=gold_spans,
                )
        return data

    def _get_data_dict_inference(self) -> Dict[str, Any]:
        """Format row data for inference NER

        In inference NER, we don't have gold spans so we only need to
        assemble the data for pred spans.

        NOTE: All spans are at the token level in this fn
        """
        data: defaultdict = defaultdict(list)

        # Loop through samples
        num_samples = len(self.ids)
        for idx in range(num_samples):
            sample_id = self.ids[idx]
            pred_spans = self.pred_spans[idx]
            pred_embs = self.pred_emb[idx]
            pred_conf_probs = self.pred_conf_prob[idx]

            # Loop through the remaining pred spans
            for pred_span, pred_emb, pred_conf_prob in zip(
                pred_spans, pred_embs, pred_conf_probs
            ):
                data = self._construct_pred_span_row(
                    data,
                    sample_id,
                    pred_span,
                    pred_emb,
                    pred_conf_prob,
                )

        return data

    def _construct_gold_span_row(
        self,
        data: DefaultDict,
        sample_id: int,
        gold_span: Dict,
        gold_emb: np.ndarray,
        gold_conf_prob: np.ndarray,
        gold_loss_prob: np.ndarray,
        gold_loss_prob_label: int,
    ) -> DefaultDict:
        span_ind = (gold_span["start"], gold_span["end"])
        data = self._construct_span_row(
            d=data,
            id=sample_id,
            start=span_ind[0],
            end=span_ind[1],
            emb=gold_emb,
            conf_prob=gold_conf_prob,
        )
        data["is_gold"].append(True)
        data["gold"].append(gold_span["label"])
        data["loss_prob"].append(gold_loss_prob)
        data["loss_prob_label"].append(gold_loss_prob_label)
        return data

    def _construct_pred_span_row(
        self,
        data: DefaultDict,
        sample_id: int,
        pred_span: Dict,
        pred_emb: np.ndarray,
        pred_conf_prob: np.ndarray,
        pred_loss_prob: Optional[np.ndarray] = None,
        pred_loss_prob_label: Optional[int] = None,
        gold_spans: Optional[List[Dict]] = None,
    ) -> DefaultDict:
        start, end = pred_span["start"], pred_span["end"]
        data = self._construct_span_row(
            d=data,
            id=sample_id,
            start=start,
            end=end,
            emb=pred_emb,
            conf_prob=pred_conf_prob,
        )
        data["is_pred"].append(True)
        data["pred"].append(pred_span["label"])

        if self.split == Split.inference and self.inference_name:
            data["inference_name"].append(self.inference_name)
        else:
            assert (
                gold_spans is not None
            ), f"gold_spans must be provided for split {self.split}"
            data["is_gold"].append(False)
            data["gold"].append("")
            # Pred only spans are known as "ghost" spans (hallucinated) or no error
            err = (
                NERErrorType.ghost_span.value
                if self._is_ghost_span(pred_span, gold_spans)
                else NERErrorType.none.value
            )
            data["galileo_error_type"].append(err)
            data["loss_prob"].append(pred_loss_prob)
            data["loss_prob_label"].append(pred_loss_prob_label)

        return data

    def _is_ghost_span(self, pred_span: Dict, gold_spans: List[Dict]) -> bool:
        """Returns if the span is a ghost span

        A ghost span is a prediction span that has no overlap with any gold span.
        A ghost span is a pred_span where either:
        1. pred_end <= gold_start
        or
        2. pred_start >= gold_end

        For all gold spans
        """
        for gold_span in gold_spans:
            pred_start, pred_end = pred_span["start"], pred_span["end"]
            gold_start, gold_end = gold_span["start"], gold_span["end"]
            is_ghost = pred_start >= gold_end or pred_end <= gold_start
            if not is_ghost:  # If we ever hit not ghost, we can fail fast
                return False
        return True

    def _construct_span_row(
        self,
        d: DefaultDict,
        id: int,
        start: int,
        end: int,
        emb: ndarray,
        conf_prob: ndarray,
    ) -> DefaultDict:
        d["sample_id"].append(id)
        d["epoch"].append(self.epoch)
        d["split"].append(Split(self.split).value)
        d["data_schema_version"].append(__data_schema_version__)
        d["span_start"].append(start)
        d["span_end"].append(end)
        d["emb"].append(emb)
        d["conf_prob"].append(conf_prob)
        return d

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
            # We compare to the end of the spans non-inclusive because
            # at the token level (which these are), the end index won't include a
            # "space" character like char-level span indices would. So a gold span
            # {start: 1, end: 5} and a pred span {start: 5, end: 9} should NOT be
            # considered a span_shift, that is a missed label
            if (
                gold_start <= pred_start < gold_end
                or pred_start <= gold_start < pred_end
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
