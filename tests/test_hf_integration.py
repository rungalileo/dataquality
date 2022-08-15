import json
from typing import Dict, List
from unittest import mock

import datasets
import pytest

from dataquality.exceptions import GalileoException
from dataquality.integrations.hf import (
    _validate_dataset,
    get_dataloader,
    infer_schema,
    tokenize_adjust_labels,
    tokenize_and_log_dataset,
)
from dataquality.schemas.ner import TaggingSchema
from dataquality.utils.hf_tokenizer import extract_gold_spans_at_word_level
from tests.utils.hf_integration_constants import (
    ADJUSTED_TOKEN_DATA,
    BILOUSequence,
    BIOESSequence,
    BIOSequence,
    mock_ds,
    mock_tokenizer,
    tag_names,
)


@pytest.mark.parametrize(
    "labels,schema",
    [
        (["B-PER", "I-PER", "B-ORG", "O"], TaggingSchema.BIO),
        (["B-PER", "I-PER", "B-ORG", "U-ORG", "L-ORG", "O"], TaggingSchema.BILOU),
        (["B-PER", "I-PER", "B-ORG", "S-ORG", "E-ORG", "O"], TaggingSchema.BIOES),
        (["G", "A", "R", "B", "A", "G", "E"], "error"),
    ],
)
def test_infer_schema(labels: List[str], schema: TaggingSchema) -> None:
    if schema == "error":
        with pytest.raises(GalileoException):
            infer_schema(labels)
    else:
        assert infer_schema(labels) == schema


def test_tokenize_adjust_labels() -> None:
    output = tokenize_adjust_labels(mock_ds, mock_tokenizer, tag_names)
    for k in ADJUSTED_TOKEN_DATA:
        assert ADJUSTED_TOKEN_DATA[k] == output[k]


@pytest.mark.parametrize(
    "gold_sequence,gold_span",
    list(zip(BIOSequence.gold_sequences, BIOSequence.gold_spans)),
)
def test_extract_gold_spans_at_word_level_bio(
    gold_sequence: List[str], gold_span: List[Dict]
) -> None:
    assert (
        extract_gold_spans_at_word_level(gold_sequence, TaggingSchema.BIO) == gold_span
    )


@pytest.mark.parametrize(
    "gold_sequence,gold_span",
    list(zip(BIOESSequence.gold_sequences, BIOESSequence.gold_spans)),
)
def test_extract_gold_spans_at_word_level_bioes(
    gold_sequence: List[str], gold_span: List[Dict]
) -> None:
    assert (
        extract_gold_spans_at_word_level(gold_sequence, TaggingSchema.BIOES)
        == gold_span
    )


@pytest.mark.parametrize(
    "gold_sequence,gold_span",
    list(zip(BILOUSequence.gold_sequences, BILOUSequence.gold_spans)),
)
def test_extract_gold_spans_at_word_level_bilou(
    gold_sequence: List[str], gold_span: List[Dict]
) -> None:
    assert (
        extract_gold_spans_at_word_level(gold_sequence, TaggingSchema.BILOU)
        == gold_span
    )


def test_validate_dataset() -> None:
    ds = datasets.Dataset.from_dict(
        dict(
            my_text=["sample1", "sample2", "sample3"],
            my_labels=["A", "A", "B"],
            my_id=[1, 2, 3],
        )
    )
    # Must be a DatasetDict, not Dataset
    with pytest.raises(GalileoException) as e:
        _validate_dataset(ds)
    assert str(e.value) == (
        f"Expected DatasetDict but got object of type {type(ds)}. "
        f"If this is a dataset, you can create a dataset dict by running\n"
        "dd = datasets.DatasetDict({'your_split': your_Dataset})"
    )


@mock.patch("dataquality.log_dataset")
def test_tokenize_and_log_dataset(mock_log_dataset: mock.MagicMock) -> None:
    """Tests the e2e function call, passing in a DatasetDict and receiving a

    new DatasetDict, and that the datasets per split were logged correctly.
    """

    tokenize_output = tokenize_adjust_labels(mock_ds, mock_tokenizer, tag_names)
    with mock.patch("dataquality.integrations.hf.tokenize_adjust_labels") as mock_tok:
        mock_tok.return_value = tokenize_output
        ds_dict = datasets.DatasetDict(
            {"train": mock_ds, "test": mock_ds, "validation": mock_ds}
        )
        output = tokenize_and_log_dataset(ds_dict, mock_tokenizer)
    for split in ds_dict.keys():
        split_output = output[split]
        for k in ADJUSTED_TOKEN_DATA:
            token_data = ADJUSTED_TOKEN_DATA[k]
            if k == "text_token_indices":
                # We abuse token data because outputs are returning tuples but we want
                # to compare lists
                token_data = json.loads(json.dumps(token_data))
            assert token_data == split_output[k]
    assert mock_log_dataset.call_count == 3


def test_get_dataloader() -> None:
    dataset = ADJUSTED_TOKEN_DATA.copy()
    dataset["id"] = list(range(len(dataset["input_ids"])))
    ds = datasets.Dataset.from_dict(dataset)
    loader = get_dataloader(ds)
    assert len(loader.dataset) == len(mock_ds)
    assert sorted(list(loader.dataset[2].keys())) == sorted(
        ["id", "input_ids", "attention_mask", "labels"]
    )
