from typing import List, Set

from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
from transformers import BatchEncoding, PreTrainedTokenizerBase

import dataquality as dq
from dataquality.exceptions import GalileoException
from dataquality.schemas.hf import HFCol
from dataquality.schemas.ner import TaggingSchema
from dataquality.schemas.split import conform_split
from dataquality.utils.hf_tokenizer import LabelTokenizer


def _is_bio(schema_tags: Set[str]) -> bool:
    return sorted(list(schema_tags)) == sorted(["B", "I", "O"])


def _is_bioes(schema_tags: Set[str]) -> bool:
    return sorted(list(schema_tags)) == sorted(["B", "I", "O", "E", "S"])


def _is_bilou(schema_tags: Set[str]) -> bool:
    return sorted(list(schema_tags)) == sorted(["B", "I", "L", "O", "U"])


def infer_schema(label_list: List[str]) -> TaggingSchema:
    """Infers the schema via the exhaustive list of labels"""
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


def tokenize_adjust_labels(
    all_samples_per_split: Dataset,
    tokenizer: PreTrainedTokenizerBase,
) -> BatchEncoding:
    tag_names = all_samples_per_split.features[HFCol.ner_tags].feature.names
    schema = infer_schema(tag_names)
    label_tokenizer = LabelTokenizer(all_samples_per_split, tokenizer, schema)

    for k in range(label_tokenizer.num_samples):
        label_tokenizer.initialize_batch(k)
        if label_tokenizer.skip_batch:
            continue

        for w_index_bpe, wid in enumerate(label_tokenizer.word_ids):
            # Logic for text_token_indices
            wid_is_none = label_tokenizer.update_text_token_indices(k, w_index_bpe, wid)
            if wid_is_none:
                continue

            # Logic to adjust labels for BPE
            num_word_gold_spans = len(label_tokenizer.word_gold_spans)
            if label_tokenizer.current_gold_span_idx != num_word_gold_spans:
                label_tokenizer.adjust_labels_bpe(wid, w_index_bpe)

        num_adjusted_labels = len(label_tokenizer.adjusted_label_indices)
        assert num_adjusted_labels == len(label_tokenizer.text_token_indices) + 2

        label_tokenizer.update_totals_for_batch(k)

    return label_tokenizer.tokenized_samples


def _validate_dataset(dd: DatasetDict) -> None:
    """Validates that the dataset passed in is a DatasetDict"""
    if not isinstance(dd, DatasetDict):
        raise GalileoException(
            f"Expected DatasetDict but got object of type {type(dd)}. "
            f"If this is a dataset, you can create a dataset dict by running"
            # TODO Add snippet
        )


def tokenize_and_log_dataset(
    ds: DatasetDict, tokenizer: PreTrainedTokenizerBase
) -> DatasetDict:
    """This function tokenizes a huggingface DatasetDict and aligns the labels to BPE

    After tokenization, this function will also log the dataset(s) present in the
    DatasetDict

    :param ds: DatasetDict from huggingface to log
    :param tokenizer: The pretrained tokenizer from huggingface
    """
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
        dataset: Dataset = tokenized_dataset[split]
        dq.log_dataset(dataset, split=dq_split)  # type: ignore
    return tokenized_dataset
