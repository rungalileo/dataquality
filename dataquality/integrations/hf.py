from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
from transformers import BatchEncoding, PreTrainedTokenizerBase

import dataquality as dq
from dataquality.exceptions import GalileoException
from dataquality.schemas.hf import HFCol
from dataquality.schemas.ner import TaggingSchema
from dataquality.schemas.split import conform_split
from dataquality.utils.hf_tokenizer import LabelTokenizer


def tokenize_adjust_labels(
    all_samples_per_split: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    schema: TaggingSchema,
) -> BatchEncoding:
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
