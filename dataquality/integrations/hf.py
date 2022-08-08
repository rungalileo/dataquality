import warnings
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
from transformers import BatchEncoding, PreTrainedTokenizerBase

import dataquality as dq
from dataquality.exceptions import GalileoException, GalileoWarning
from dataquality.loggers.model_logger.text_ner import TextNERModelLogger
from dataquality.schemas.ner import TaggingSchema
from dataquality.schemas.split import conform_split


@dataclass
class HFCol:
    input_ids: str = "input_ids"
    text: str = "text"
    id: str = "id"
    ner_tags: str = "ner_tags"
    text_token_indices: str = "text_token_indices"
    tokens: str = "tokens"
    bpe_tokens: str = "bpe_tokens"
    gold_spans: str = "gold_spans"
    label: str = "label"
    start: str = "start"
    end: str = "end"


class LabelTokenizer:
    def __init__


def _is_bio(schema_tags: Set[str]) -> bool:
    return sorted(list(schema_tags)) == sorted(["B", "I", "O"])


def _is_bioes(schema_tags: Set[str]) -> bool:
    return sorted(list(schema_tags)) == sorted(["B", "I", "O", "E", "S"])


def _is_bilou(schema_tags: Set[str]) -> bool:
    return sorted(list(schema_tags)) == sorted(["B", "I", "L", "O", "U"])


def infer_schema(label_list: List[str]) -> TaggingSchema:
    schema_tags = set([x.split("-")[0] for x in label_list])
    if _is_bio(schema_tags):
        return TaggingSchema.BIO
    elif _is_bioes(schema_tags):
        return TaggingSchema.BIOES
    elif _is_bilou(schema_tags):
        return TaggingSchema.BILOU
    else:
        raise GalileoException(
            "Tagging schema must be one of BIO, BIOES, or BILOU. Given schemas tags "
            f"{schema_tags} we cannot identify the tagging schema."
        )


def extract_gold_spans_at_word_level(gold_sequence: List[str]) -> List[Dict]:
    """Extracts span level words from a gold sequence

    Given a gold sequence [O, O, B-PER, I-PER, I-PER, ...] -> extracts out spans
    at the word level

    gold_sequence = ['B-LOC', 'I-LOC', 'I-LOC', 'I-LOC', 'I-LOC']
    gold_spans = extract_gold_spans_at_word_level(gold_sequence)
    # gold_spans -> [{'start': 0, 'end': 5, 'label': 'LOC'}]
    """
    logger = TextNERModelLogger()
    schema = infer_schema(gold_sequence)
    if schema == TaggingSchema.BIO:
        return logger._extract_spans_bio(gold_sequence)
    else:  # BILOU or BIOES
        return logger._extract_spans_token_level(gold_sequence)


def tokenize_adjust_labels(
    all_samples_per_split: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    schema: TaggingSchema,
) -> BatchEncoding:
    tokenized_samples = tokenizer.batch_encode_plus(
        all_samples_per_split[HFCol.tokens], is_split_into_words=True
    )

    total_adjusted_labels = []
    total_text_token_indices = []
    total_bpe_tokens = []
    texts = []  # TODO: Add warning that we assume space seperation here
    idx_2_labels = all_samples_per_split.features[HFCol.ner_tags].feature.names
    labels_2_idx = {k: v for v, k in enumerate(idx_2_labels)}  # TODO: make this k:v
    total_gold_spans = []

    for k in range(0, len(tokenized_samples[HFCol.input_ids])):
        prev_word_id = -1
        word_ids_list = tokenized_samples.word_ids(batch_index=k)
        existing_labels = [
            idx_2_labels[label] for label in all_samples_per_split[HFCol.ner_tags][k]
        ]
        word_gold_spans = extract_gold_spans_at_word_level(existing_labels)
        if len(word_gold_spans) == 0:
            warnings.warn(
                f"No gold spans found for batch {k}. This batch will not be logged",
                GalileoWarning
            )
            continue
        original_word_idx = -1
        char_seen = -len(all_samples_per_split[HFCol.tokens][k][0])
        adjusted_label_indices = [labels_2_idx["O"]] * len(word_ids_list)
        adjusted_labels = ["O"] * len(word_ids_list)
        text_token_indices = []
        start_char_idx, end_char_idx = 0, 0
        gold_spans = []  # dictionary holding character level gold spans
        current_gold_span_idx = 0

        for w_index_bpe, wid in enumerate(word_ids_list):
            # Logic for text_token_indices
            if wid is None:
                # adjusted_labels[w_index_bpe] = -100
                # TODO consider uncommenting
                # TODO write test functions for each function
                """if w_index_bpe == 0:
                  text_token_indices.append((0, 0)) # This is the CLS and SEP tokens
                else:
                  text_token_indices.append((end_char_idx, end_char_idx)) # This is the CLS and SEP tokens"""
                char_seen += len(all_samples_per_split[HFCol.tokens][k][0])
                continue
            elif wid != prev_word_id:
                original_word_idx = original_word_idx + 1
                prev_word_id = wid
                start_char_idx, end_char_idx = char_seen, char_seen + len(
                    all_samples_per_split[HFCol.tokens][k][original_word_idx]
                )
                text_token_indices.append(
                    (start_char_idx, end_char_idx)
                )  # Get the char start and end index for the word
                char_seen += len(all_samples_per_split[HFCol.tokens][k][original_word_idx]) + 1
            else:
                text_token_indices.append(
                    (start_char_idx, end_char_idx)
                )  # Get the char start and end index for the word

            # Logic to adjust labels for BPE
            if current_gold_span_idx != len(word_gold_spans):
                span_start_word = word_gold_spans[current_gold_span_idx][HFCol.start]
                span_end_word = word_gold_spans[current_gold_span_idx][HFCol.end] - 1
                span_label_suffix = word_gold_spans[current_gold_span_idx][HFCol.label]
                # Recognized a singelton length span that could be in start, middle or end of sentence
                if (
                    (span_start_word == span_end_word)
                    and (wid == span_start_word)
                    and (
                        wid != word_ids_list[w_index_bpe - 1]
                        and wid != word_ids_list[w_index_bpe + 1]
                    )
                ):
                    if schema == TaggingSchema.BILOU:
                        adjusted_labels[w_index_bpe] = f"U-{span_label_suffix}"
                    if schema == TaggingSchema.BIOES:
                        adjusted_labels[w_index_bpe] = f"S-{span_label_suffix}"
                    else:
                        adjusted_labels[w_index_bpe] = f"B-{span_label_suffix}"
                    adjusted_label_indices[w_index_bpe] = labels_2_idx[
                        adjusted_labels[w_index_bpe]
                    ]
                    current_gold_span_idx += 1
                    gold_spans.append(
                        {
                            HFCol.start: start_char_idx,
                            HFCol.end: end_char_idx,
                            HFCol.label: span_label_suffix,
                        }
                    )

                # Recognized within a span
                elif wid >= span_start_word and wid <= span_end_word:
                    # Hit the start of a span and start BPEs
                    if (w_index_bpe == 0) or (
                        wid != word_ids_list[w_index_bpe - 1] and wid == span_start_word
                    ):  # first BPE
                        adjusted_labels[w_index_bpe] = f"B-{span_label_suffix}"
                        adjusted_label_indices[w_index_bpe] = labels_2_idx[
                            adjusted_labels[w_index_bpe]
                        ]
                        gold_spans.append(
                            {
                                HFCol.start: start_char_idx,
                                HFCol.label: span_label_suffix,
                            }
                        )
                    elif (w_index_bpe == len(word_ids_list) - 1) or (
                        wid != word_ids_list[w_index_bpe + 1] and wid == span_end_word
                    ):  # last BPE
                        if schema == TaggingSchema.BILOU:
                            adjusted_labels[w_index_bpe] = f"L-{span_label_suffix}"
                        if schema == TaggingSchema.BIOES:
                            adjusted_labels[w_index_bpe] = f"E-{span_label_suffix}"
                        else:
                            adjusted_labels[w_index_bpe] = f"I-{span_label_suffix}"
                        adjusted_label_indices[w_index_bpe] = labels_2_idx[
                            adjusted_labels[w_index_bpe]
                        ]
                        gold_spans[-1][HFCol.end] = end_char_idx  # Update end indices
                        current_gold_span_idx += 1
                    else:  # other BPEs
                        adjusted_labels[w_index_bpe] = f"I-{span_label_suffix}"
                        adjusted_label_indices[w_index_bpe] = labels_2_idx[
                            adjusted_labels[w_index_bpe]
                        ]
        assert len(adjusted_label_indices) == len(text_token_indices) + 2

        total_adjusted_labels.append(adjusted_label_indices)
        total_text_token_indices.append(text_token_indices)
        total_bpe_tokens.append(tokenized_samples[k].tokens)
        texts.append(" ".join(all_samples_per_split[HFCol.tokens][k]))
        total_gold_spans.append(gold_spans)

    tokenized_samples[HFCol.label] = total_adjusted_labels
    tokenized_samples[HFCol.text_token_indices] = total_text_token_indices
    tokenized_samples[HFCol.bpe_tokens] = total_bpe_tokens
    tokenized_samples[HFCol.text] = texts
    tokenized_samples[HFCol.gold_spans] = total_gold_spans

    return tokenized_samples


def _validate_dataset(ds: DatasetDict) -> None:
    """Validates that the dataset passed in is a DatasetDict"""
    if not isinstance(ds, DatasetDict):
        raise GalileoException(
            f"Expected DatasetDict but got object of type {type(ds)}. "
            f"If this is a dataset, you can create a dataset dict by running"
            # TODO Add snippet
        )


def tokenize_and_align_labels(ds: DatasetDict, tokenizer: PreTrainedTokenizerBase):
    _validate_dataset(ds)
    tokenized_dataset = ds.map(
        tokenize_adjust_labels,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer},
    )
    splits = tokenized_dataset.keys()

    for split in splits:
        dq_split = conform_split(split)
        ids = list(range(len(tokenized_dataset[split])))
        tokenized_dataset[split] = tokenized_dataset[split].add_column(HFCol.id, ids)
        dq.log_dataset(tokenized_dataset[split], split=dq_split)
    return tokenized_dataset
