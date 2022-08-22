from typing import Dict, List, Tuple

from datasets import Dataset
from transformers import PreTrainedTokenizerBase

from dataquality.loggers.model_logger.text_ner import TextNERModelLogger
from dataquality.schemas.hf import HFCol, SpanKey
from dataquality.schemas.ner import TaggingSchema


def extract_gold_spans_at_word_level(
    gold_sequence: List[str], schema: TaggingSchema
) -> List[Dict]:
    """Extracts span level words from a gold sequence

    Given a gold sequence [O, O, B-PER, I-PER, I-PER, ...] -> extracts out spans
    at the word level

    gold_sequence = ['B-LOC', 'I-LOC', 'I-LOC', 'I-LOC', 'I-LOC']
    gold_spans = extract_gold_spans_at_word_level(gold_sequence)
    # gold_spans -> [{'start': 0, 'end': 5, 'label': 'LOC'}]
    """
    logger = TextNERModelLogger()
    if schema == TaggingSchema.BIO:
        return logger._extract_spans_bio(gold_sequence)
    else:  # BILOU or BIOES
        return logger._extract_spans_token_level(gold_sequence)


class LabelTokenizer:
    """
    Class that allows Galileo users to directly tokenize their data using a provided HF
    tokenizer, infers the schema based on provided labels, and also align the labels
    for the tokenized data.
    Galileo automatically extracts out the required input data and logs it.
    """

    def __init__(
        self,
        ds: Dataset,
        tokenizer: PreTrainedTokenizerBase,
        schema: TaggingSchema,
        label_names: List[str],
    ) -> None:
        self.ds = ds
        self.schema = schema
        self.tokenized_samples = tokenizer.batch_encode_plus(
            ds[HFCol.tokens], is_split_into_words=True
        )

        self.total_adjusted_labels_indices: List[List[int]] = []
        self.total_text_token_indices: List[List[Tuple]] = []
        self.total_bpe_tokens: List[List[str]] = []
        self.texts: List[str] = []
        self.idx_2_labels = label_names
        self.labels_2_idx = {k: v for v, k in enumerate(self.idx_2_labels)}
        self.total_gold_spans: List[List[Dict]] = []
        self.num_samples = len(self.tokenized_samples[HFCol.input_ids])

        # Batch initialization
        self.previous_word_id = -1
        self.word_ids: List[int] = []
        self.word_gold_spans: List[Dict] = []
        self.original_word_idx = -1
        self.char_seen = -1
        self.adjusted_label_indices: List[int] = []
        self.adjusted_labels: List[str] = []
        self.text_token_indices: List[Tuple] = []
        self.start_char_idx = -1
        self.end_char_idx = -1
        self.gold_spans: List[Dict] = []
        self.current_gold_span_idx = -1

    def initialize_sample(self, k: int) -> None:
        self.previous_word_id = -1
        self.word_ids = self.tokenized_samples.word_ids(batch_index=k)
        existing_labels = [
            self.idx_2_labels[label] for label in self.ds[HFCol.ner_tags][k]
        ]
        self.word_gold_spans = extract_gold_spans_at_word_level(
            existing_labels, self.schema
        )
        self.original_word_idx = -1
        self.char_seen = -len(self.ds[HFCol.tokens][k][0])
        self.adjusted_label_indices = [self.labels_2_idx["O"]] * len(self.word_ids)
        self.adjusted_labels = ["O"] * len(self.word_ids)
        self.text_token_indices = []
        self.start_char_idx, self.end_char_idx = 0, 0
        self.gold_spans = []
        self.current_gold_span_idx = 0

    def update_text_token_indices(self, k: int, w_index_bpe: int, wid: int) -> bool:
        if wid is None:
            self.char_seen += len(self.ds[HFCol.tokens][k][0])
            return True
        elif wid != self.previous_word_id:
            self.original_word_idx = self.original_word_idx + 1
            self.previous_word_id = wid
            self.start_char_idx = self.char_seen
            self.end_char_idx = self.char_seen + len(
                self.ds[HFCol.tokens][k][self.original_word_idx]
            )
            # Get the char start and end index for the word
            self.text_token_indices.append((self.start_char_idx, self.end_char_idx))
            self.char_seen += len(self.ds[HFCol.tokens][k][self.original_word_idx]) + 1
        else:
            # Get the char start and end index for the word
            self.text_token_indices.append((self.start_char_idx, self.end_char_idx))
        # Word ID is not None
        return False

    def _is_singleton_span(
        self, wid: int, w_index_bpe: int, span_start_word: int, span_end_word: int
    ) -> bool:
        return (
            span_start_word == span_end_word
            and wid == span_start_word
            and wid != self.word_ids[w_index_bpe - 1]
            and wid != self.word_ids[w_index_bpe + 1]
        )

    def _is_within_span(
        self, wid: int, span_start_word: int, span_end_word: int
    ) -> bool:
        return span_start_word <= wid <= span_end_word

    def _is_first_bpe(self, wid: int, w_index_bpe: int, span_start_word: int) -> bool:
        is_first_index = w_index_bpe == 0
        wid_is_first = wid != self.word_ids[w_index_bpe - 1] and wid == span_start_word
        return is_first_index or wid_is_first

    def _is_last_bpe(self, wid: int, w_index_bpe: int, span_end_word: int) -> bool:
        is_last_index = w_index_bpe == len(self.word_ids) - 1
        wid_is_last = wid != self.word_ids[w_index_bpe + 1] and wid == span_end_word
        return is_last_index or wid_is_last

    def _adjust_label_at_index(self, w_index_bpe: int, span_label_suffix: str) -> None:
        if self.schema == TaggingSchema.BILOU:
            self.adjusted_labels[w_index_bpe] = f"U-{span_label_suffix}"
        if self.schema == TaggingSchema.BIOES:
            self.adjusted_labels[w_index_bpe] = f"S-{span_label_suffix}"
        else:
            self.adjusted_labels[w_index_bpe] = f"B-{span_label_suffix}"
        self.adjusted_label_indices[w_index_bpe] = self.labels_2_idx[
            self.adjusted_labels[w_index_bpe]
        ]
        self.current_gold_span_idx += 1
        self.gold_spans.append(
            {
                SpanKey.start: self.start_char_idx,
                SpanKey.end: self.end_char_idx,
                SpanKey.label: span_label_suffix,
            }
        )

    def _adjust_first_bpe(self, w_index_bpe: int, span_label_suffix: str) -> None:
        self.adjusted_labels[w_index_bpe] = f"B-{span_label_suffix}"
        self.adjusted_label_indices[w_index_bpe] = self.labels_2_idx[
            self.adjusted_labels[w_index_bpe]
        ]
        self.gold_spans.append(
            {
                SpanKey.start: self.start_char_idx,
                SpanKey.label: span_label_suffix,
            }
        )

    def _adjust_last_bpe(self, w_index_bpe: int, span_label_suffix: str) -> None:
        if self.schema == TaggingSchema.BILOU:
            self.adjusted_labels[w_index_bpe] = f"L-{span_label_suffix}"
        if self.schema == TaggingSchema.BIOES:
            self.adjusted_labels[w_index_bpe] = f"E-{span_label_suffix}"
        else:
            self.adjusted_labels[w_index_bpe] = f"I-{span_label_suffix}"
        self.adjusted_label_indices[w_index_bpe] = self.labels_2_idx[
            self.adjusted_labels[w_index_bpe]
        ]
        # Update end indices
        self.gold_spans[-1][SpanKey.end] = self.end_char_idx
        self.current_gold_span_idx += 1

    def _adjust_middle_bpe(self, w_index_bpe: int, span_label_suffix: str) -> None:
        self.adjusted_labels[w_index_bpe] = f"I-{span_label_suffix}"
        self.adjusted_label_indices[w_index_bpe] = self.labels_2_idx[
            self.adjusted_labels[w_index_bpe]
        ]

    def adjust_labels_bpe(self, wid: int, w_index_bpe: int) -> None:
        span_start_word = self.word_gold_spans[self.current_gold_span_idx][
            SpanKey.start
        ]
        span_end_word = (
            self.word_gold_spans[self.current_gold_span_idx][SpanKey.end] - 1
        )
        span_label_sfx = self.word_gold_spans[self.current_gold_span_idx][SpanKey.label]
        # Found a singelton length span that could be in the
        # start, middle, or end of the sentence
        if self._is_singleton_span(wid, w_index_bpe, span_start_word, span_end_word):
            self._adjust_label_at_index(w_index_bpe, span_label_sfx)

        # Recognized within a span
        elif self._is_within_span(wid, span_start_word, span_end_word):
            # Hit the start of a span and start BPEs
            if self._is_first_bpe(wid, w_index_bpe, span_start_word):
                self._adjust_first_bpe(w_index_bpe, span_label_sfx)
            elif self._is_last_bpe(wid, w_index_bpe, span_end_word):
                self._adjust_last_bpe(w_index_bpe, span_label_sfx)
            else:  # other BPEs
                self._adjust_middle_bpe(w_index_bpe, span_label_sfx)

    def update_totals_for_sample(self, k: int) -> None:
        self.total_adjusted_labels_indices.append(self.adjusted_label_indices)
        self.total_text_token_indices.append(self.text_token_indices)
        self.total_bpe_tokens.append(self.tokenized_samples[k].tokens)
        self.texts.append(" ".join(self.ds[HFCol.tokens][k]))
        self.total_gold_spans.append(self.gold_spans)

    def update_tokenized_samples(self) -> None:
        self.tokenized_samples[HFCol.labels] = self.total_adjusted_labels_indices
        self.tokenized_samples[HFCol.text_token_indices] = self.total_text_token_indices
        self.tokenized_samples[HFCol.bpe_tokens] = self.total_bpe_tokens
        self.tokenized_samples[HFCol.text] = self.texts
        self.tokenized_samples[HFCol.gold_spans] = self.total_gold_spans
