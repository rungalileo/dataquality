import warnings
from collections import defaultdict
from enum import Enum, unique
from typing import Any, DefaultDict, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy import ndarray

from dataquality.loggers.logger_config.text_ner import text_ner_logger_config
from dataquality.loggers.model_logger.base_model_logger import BaseGalileoModelLogger
from dataquality.schemas import __data_schema_version__
from dataquality.schemas.ner import NERErrorType, TaggingSchema
from dataquality.schemas.split import Split


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
    logits = "logits"
    ids = "ids"
    # mixin restriction on str (due to "str".split(...))
    split = "split"  # type: ignore
    epoch = "epoch"
    dep_scores = "dep_scores"
    log_helper_data = "log_helper_data"
    inference_name = "inference_name"

    @staticmethod
    def get_valid() -> List[str]:
        return list(map(lambda x: x.value, GalileoModelLoggerAttributes))


class TextNERModelLogger(BaseGalileoModelLogger):
    """
    Class for logging model output data of Text NER models to Galileo.

    * emb: List[np.ndarray]: Each np.ndarray represents all embeddings of a given
    sample. These embeddings are from the tokenized text, and will align with the tokens
    in the sample. If you have 12 samples in the dataset, with each sample of 20 tokens
    in length, and an embedding vector of size 768, len(emb) will be 12, and
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
            [np.array([embs(the), embs(president), embs(is), embs(joe),
            embs(bi), embs(##den), embs(<pad>), embs(<pad>), embs(<pad>)]),
            np.array([embs(joe), embs(bi), embs(##den), embs(addressed),
            embs(the), embs(united), embs(states), embs(on), embs(monday)])]

        epoch = 0
        ids = [0, 1]  # Must match the data input IDs
        split = "training"
        dataquality.log_model_outputs(
            emb=emb, logits=logits, ids=ids, split=split, epoch=epoch
        )
    """

    __logger_name__ = "text_ner"
    logger_config = text_ner_logger_config

    def __init__(
        self,
        emb: List[np.ndarray] = None,
        probs: List[np.ndarray] = None,
        logits: List[np.ndarray] = None,
        ids: Union[List, np.ndarray] = None,
        split: Optional[str] = None,
        epoch: Optional[int] = None,
    ) -> None:
        super().__init__()
        # Need to compare to None because they may be np arrays which cannot be
        # evaluated with bool directly
        self.emb = emb if emb is not None else []
        self.probs = probs if probs is not None else []
        self.logits = logits if logits is not None else []
        self.ids: Union[List, np.ndarray] = ids if ids is not None else []
        self.split = split
        self.epoch = epoch

        # Calculated internally
        self.gold_emb: List[List[np.ndarray]] = []
        self.gold_spans: List[List[Dict]] = []
        self.gold_dep: List[List[float]] = []

        self.pred_emb: List[List[np.ndarray]] = []
        self.pred_spans: List[List[Dict]] = []
        self.pred_dep: List[List[float]] = []

        # Used for helper data, does not get logged
        self.log_helper_data: Dict[str, Any] = {}

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

        if len(self.logits):
            self.probs = self.convert_logits_to_probs(self.logits).tolist()
        elif len(self.probs):
            warnings.warn("Usage of probs is deprecated, use logits instead")

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
            # This will return True if there was a prediction or gold span
            if self._process_sample(sample_id, sample_emb, sample_prob):
                logged_sample_ids.append(sample_id)

        self.ids = logged_sample_ids
        assert self.ids, (
            "No samples in this batch had any gold or prediction spans. Logging will "
            "be skipped"
        )

    def _process_sample(
        self, sample_id: int, sample_emb: np.ndarray, sample_prob: np.ndarray
    ) -> bool:
        """Processes a sample. Returns whether or not the sample should be logged

        A sample should be logged only if there was at least 1 prediction span or 1
        gold span
        """
        # To extract metadata about the sample we are looking at
        sample_key = self.logger_config.get_sample_key(Split(self.split), sample_id)

        # Unpadded length of the sample. Used to extract true predicted spans
        # which are padded by the model
        sample_token_len = self.logger_config.sample_length[sample_key]
        # Get prediction spans
        sample_pred_spans = self._extract_pred_spans(sample_prob, sample_token_len)
        # Get gold (ground truth) spans
        gold_span_tup = self.logger_config.gold_spans.get(sample_key, [])
        sample_gold_spans: List[Dict] = [
            dict(start=start, end=end, label=label)
            for start, end, label in gold_span_tup
        ]
        # If there were no golds and no preds for a sample, don't log this sample
        if not sample_pred_spans and not sample_gold_spans:
            return False

        gold_dep, pred_dep = self._calculate_dep_scores(
            sample_prob, sample_gold_spans, sample_pred_spans, sample_token_len
        )
        gold_emb = self._extract_span_embeddings(sample_gold_spans, sample_emb)
        pred_emb = self._extract_span_embeddings(sample_pred_spans, sample_emb)

        self.pred_spans.append(sample_pred_spans)
        self.gold_spans.append(sample_gold_spans)
        self.pred_dep.append(pred_dep)
        self.gold_dep.append(gold_dep)
        self.gold_emb.append(gold_emb)
        self.pred_emb.append(pred_emb)
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

    def _extract_pred_spans(self, pred_prob: np.ndarray, sample_len: int) -> List[Dict]:
        """
        Extract prediction labels from probabilities, and generate pred spans

        If the schema is non-BIO, we just first convert them into BIO then extract spans
        """
        # use length of the tokens stored to strip the pads
        # Drop the spans post first PAD
        argmax_indices: List[int] = np.array(pred_prob).argmax(axis=1)
        pred_sequence: List[str] = [
            self.logger_config.labels[x] for x in argmax_indices
        ][0:sample_len]

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

    def _extract_pred_spans_bilou(self, pred_sequence: List[str]) -> List[Dict]:
        """BILOU is a special case for BIO.

        The presense of I in a sequence does not mean a
        presence of a span until an L is successfully predicted.
        """
        pred_spans = []
        total_b_count = 0
        idx = 0
        found_end = False  # Tracks if there was an end tag predicted for BIOES

        # Use a while loop so we can skip rows already scanned in the inner loop
        while idx < len(pred_sequence):
            token = pred_sequence[idx]
            next_idx = idx + 1

            if self._is_single_token(token):
                total_b_count += 1
                # hit a single token prediction, update and continue
                token_val, token_label = self._split_token(token)
                pred_spans.append({"start": idx, "end": next_idx, "label": token_label})
                idx += 1
                continue

            if not self._is_before_token(token):
                idx += 1
                continue

            # We've hit a prediction. Continue until it's invalid
            # Invalid means we hit a new B or O tag, or the next I tag has a
            # different label
            token_val, token_label = self._split_token(token)

            for next_tok in pred_sequence[idx + 1 :]:
                if self._is_in_token(next_tok, token_label):
                    next_idx += 1
                # next_tok == "L" and the label matches the current B label
                elif self._is_end_token(next_tok, token_label):
                    next_idx += 1
                    found_end = True
                    total_b_count += 1
                    break
                else:
                    break
            if found_end:
                pred_spans.append({"start": idx, "end": next_idx, "label": token_label})
            idx = next_idx
            found_end = False

        assert total_b_count == len(pred_spans)
        return pred_spans

    def _extract_pred_spans_bioes(self, pred_sequence: List[str]) -> List[Dict]:
        """BIOES is a special case for BIO.

        The presense of I in a sequence does not mean a presence of a span until
        an E is successfully predicted.
        """
        pred_spans = []
        total_b_count = 0
        idx = 0
        found_end = False  # Tracks if there was an end tag predicted for BIOES

        # Use a while loop so we can skip rows already scanned in the inner loop
        while idx < len(pred_sequence):
            token = pred_sequence[idx]
            next_idx = idx + 1

            if self._is_single_token(token):
                total_b_count += 1
                # hit a single token prediction , update and continue
                token_val, token_label = self._split_token(token)
                pred_spans.append({"start": idx, "end": next_idx, "label": token_label})
                idx += 1
                continue

            if not self._is_before_token(token):
                idx += 1
                continue

            # We've hit a prediction. Continue until it's invalid
            # Invalid means we hit a new B or O tag, or the next I tag has a
            # different label
            token_val, token_label = self._split_token(token)

            for next_tok in pred_sequence[idx + 1 :]:
                # next_tok == "I" and the label matches the current B label
                if self._is_in_token(next_tok, token_label):
                    next_idx += 1
                # next_tok == "E" and the label matches the current B label
                elif self._is_end_token(next_tok, token_label):
                    next_idx += 1
                    found_end = True
                    total_b_count += 1
                    break
                else:
                    break
            if found_end:
                pred_spans.append({"start": idx, "end": next_idx, "label": token_label})
            idx = next_idx
            found_end = False

        assert total_b_count == len(pred_spans)
        return pred_spans

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

    def _calculate_dep_score_across_spans(
        self, spans: List[Dict], dep_scores: List[float]
    ) -> List[float]:
        """Computes dep score for all spans in a sample

        spans: All spans in a given sample
        dep_scores: DEP scores for every token in a sample, so len(dep_scores) is
            the number of tokens in a sentence
        """
        dep_score_per_span = []
        for span in spans:
            start = span["start"]
            end = span["end"]
            dep_score_per_span.append(max(dep_scores[start:end]))
        assert len(dep_score_per_span) == len(spans)
        return dep_score_per_span

    def _calculate_dep_scores(
        self,
        pred_prob: np.ndarray,
        gold_spans: List[Dict],
        pred_spans: List[Dict],
        sample_token_len: int,
    ) -> Tuple[List[float], List[float]]:
        """Calculates dep scores for every span in a sample

        Compute DEP score for each token using gold and predicted sequences.
        Extract DEP score for each span using max of these token-level scores
        :param gold_spans: gold spans for a sample
        :param pred_spans: predicted spans for a sample
        :param pred_prob: seq_len x num_labels probability predictions for
            every token in a sample
        :param sample_token_len:  Unpadded length of the sample. Used to extract true
            predicted spans which are padded by the model
        :return: The DEP score per-token for both the gold spans and pred spans
        """
        pred_prob = np.array(pred_prob)
        label2idx = {l: i for i, l in enumerate(self.logger_config.labels)}
        argmax_indices: List[int] = pred_prob.argmax(axis=1).tolist()
        pred_sequence: List[str] = [
            self.logger_config.labels[x] for x in argmax_indices
        ][0:sample_token_len]
        gold_sequence = self._construct_gold_sequence(len(pred_sequence), gold_spans)
        # Store dep scores for every token in the sample
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
            assert 1.0 >= dep >= 0.0, f"DEP score is out of bounds with value {dep}"
            dep_scores_tokens.append(dep)
        assert sample_token_len == len(
            dep_scores_tokens
        ), "misalignment between total tokens and DEP scores"

        # Compute dep score of each span, which is effectively max dep score across
        # all tokens in a span
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
            for gold_span, gold_dep, gold_emb in zip(gold_spans, gold_deps, gold_embs):
                data = self._construct_gold_span_row(
                    data, sample_id, gold_span, gold_emb, gold_dep
                )

                span_ind = (gold_span["start"], gold_span["end"])
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
                    data["pred"].append("")
                    error_type = self._get_span_error_type(
                        pred_span_inds, span_ind
                    ).value
                    data["galileo_error_type"].append(error_type)

            # Loop through the remaining pred spans
            for pred_span, pred_emb, pred_dep in zip(pred_spans, pred_embs, pred_deps):
                data = self._construct_pred_span_row(
                    data, sample_id, pred_span, pred_emb, pred_dep, gold_spans
                )
        return data

    def _construct_gold_span_row(
        self,
        data: DefaultDict,
        sample_id: int,
        gold_span: Dict,
        gold_emb: np.ndarray,
        gold_dep: float,
    ) -> DefaultDict:
        span_ind = (gold_span["start"], gold_span["end"])
        data = self._construct_span_row(
            data, sample_id, span_ind[0], span_ind[1], gold_dep, gold_emb
        )
        data["is_gold"].append(True)
        data["gold"].append(gold_span["label"])
        return data

    def _construct_pred_span_row(
        self,
        data: DefaultDict,
        sample_id: int,
        pred_span: Dict,
        pred_emb: np.ndarray,
        pred_dep: float,
        gold_spans: List[Dict],
    ) -> DefaultDict:
        start, end = pred_span["start"], pred_span["end"]
        data = self._construct_span_row(data, sample_id, start, end, pred_dep, pred_emb)
        data["is_gold"].append(False)
        data["is_pred"].append(True)
        data["pred"].append(pred_span["label"])
        data["gold"].append("")
        # Pred only spans are known as "ghost" spans (hallucinated) or no error
        err = (
            NERErrorType.ghost_span.value
            if self._is_ghost_span(pred_span, gold_spans)
            else NERErrorType.none.value
        )
        data["galileo_error_type"].append(err)
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
        self, d: DefaultDict, id: int, start: int, end: int, dep: float, emb: ndarray
    ) -> DefaultDict:
        d["sample_id"].append(id)
        d["epoch"].append(self.epoch)
        d["split"].append(Split(self.split).value)
        d["data_schema_version"].append(__data_schema_version__)
        d["span_start"].append(start)
        d["span_end"].append(end)
        d["data_error_potential"].append(dep)
        d["emb"].append(emb)
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

    def convert_logits_to_probs(
        self, sample_logits: Union[List, np.ndarray]
    ) -> np.ndarray:
        """Converts logits to probs via softmax per sample"""
        # axis ensures that in a matrix of probs with dims num_samples x num_classes
        # we take the softmax for each sample
        token_probs = []
        for token_logits in sample_logits:
            token_probs.append(super().convert_logits_to_probs(token_logits))
        return np.array(token_probs, dtype=object)
