from typing import Any, Dict, List, Set

from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
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
    tag_names: List[str],
) -> BatchEncoding:
    schema = infer_schema(tag_names)
    label_tokenizer = LabelTokenizer(
        all_samples_per_split, tokenizer, schema, tag_names
    )

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

    label_tokenizer.update_tokenized_samples()
    return label_tokenizer.tokenized_samples


def _validate_dataset(dd: DatasetDict) -> None:
    """Validates that the dataset passed in is a DatasetDict"""
    if not isinstance(dd, DatasetDict):
        raise GalileoException(
            f"Expected DatasetDict but got object of type {type(dd)}. "
            f"If this is a dataset, you can create a dataset dict by running\n"
            "dd = datasets.DatasetDict({'your_split': your_Dataset})"
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
    ds_keys = list(ds.keys())
    tag_names = ds[ds_keys[0]].features[HFCol.ner_tags].feature.names
    dq.set_tagging_schema(infer_schema(tag_names))
    dq.set_labels_for_run(tag_names)

    tokenized_dataset = ds.map(
        tokenize_adjust_labels,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer, "tag_names": tag_names},
    )
    splits = tokenized_dataset.keys()

    for split in splits:
        dq_split = conform_split(split)
        ids = list(range(len(tokenized_dataset[split])))
        tokenized_dataset[split] = tokenized_dataset[split].add_column(HFCol.id, ids)
        dataset: Dataset = tokenized_dataset[split]
        dq.log_dataset(dataset, split=dq_split)  # type: ignore
    return tokenized_dataset


class TextDataset(TorchDataset):
    """An abstracted Huggingface Text dataset for users to import and use

    Get back a DataLoader via the get_dataloader function"""

    def __init__(self, hf_dataset: Dataset) -> None:
        self.dataset = hf_dataset

    def __getitem__(self, idx: int) -> Dict:
        row = self.dataset[idx]
        return {
            "id": row["id"],
            "input_ids": row["input_ids"],
            "attention_mask": row["attention_mask"],
            "labels": row["labels"],
        }

    def __len__(self) -> int:
        return len(self.dataset)


def get_dataloader(dataset: Dataset, **kwargs: Any) -> DataLoader:
    """Create a DataLoader for a particular split given a huggingface Dataset

    The DataLoader will be a loader of a TextDataset. The __getitem__ for that dataset
    will return:
     * id - the Galileo ID of the sample
     * input_ids - the standard huggingface input_ids
     * attention_mask - the standard huggingface attention_mask
     * labels - output labels adjusted with tokenized NER data

    :param dataset: The huggingface dataset to convert to a DataLoader
    :param kwargs: Any additional keyword arguments to be passed into the DataLoader
        Things like batch_size or shuffle
    """
    text_dataset = TextDataset(dataset)
    return DataLoader(text_dataset, **kwargs)
