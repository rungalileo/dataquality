from typing import Dict, List
from unittest import mock

import datasets
import pytest
from transformers import BatchEncoding

from dataquality.exceptions import GalileoException
from dataquality.integrations.hf import (
    _validate_dataset,
    infer_schema,
    tokenize_adjust_labels,
    tokenize_and_log_dataset,
)
from dataquality.schemas.ner import TaggingSchema
from dataquality.utils.hf_tokenizer import extract_gold_spans_at_word_level
from tests.utils.hf_integration_constants import (
    ADJUSTED_TOKEN_DATA,
    TOKENIZED_DATA,
    UNADJUSTED_TOKEN_DATA,
    BILOUSequence,
    BIOESSequence,
    BIOSequence,
    WORD_IDS, BATCH_TOKENS, BatchEncodingTokens
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
    tokenizer = mock.Mock()
    batch_encoded = BatchEncoding(data=TOKENIZED_DATA)
    batch_encoded.word_ids = lambda batch_index: WORD_IDS[batch_index]
    batch_encoded._encodings = [BatchEncodingTokens(tokens) for tokens in BATCH_TOKENS]
    tokenizer.batch_encode_plus.return_value = batch_encoded

    ds_input = datasets.Dataset.from_dict(UNADJUSTED_TOKEN_DATA)
    ds_input.features["ner_tags"].feature.names = [
        "O",
        "B-PER",
        "I-PER",
        "B-ORG",
        "I-ORG",
        "B-LOC",
        "I-LOC",
    ]
    output = tokenize_adjust_labels(ds_input, tokenizer)
    print(output.keys())
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
    with pytest.raises(GalileoException):
        _validate_dataset(ds)


@mock.patch("dataquality.integrations.hf.dq")
def test_tokenize_and_and_log_dataset(mock_dq: mock.MagicMock) -> None:
    """TODO @ben - to test for dq logging
    Tests the e2e function call, passing in a DatasetDict and receiving a
    new DatasetDict, and that the datasets per split were logged correctly.
    """
    tokenizer = mock.Mock()
    tokenizer.batch_encode_plus.return_value = TOKENIZED_DATA
    assert ADJUSTED_TOKEN_DATA == tokenize_and_log_dataset(
        UNADJUSTED_TOKEN_DATA, tokenizer
    )
