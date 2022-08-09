from typing import Dict, List
from unittest import mock

import datasets
import pytest

from dataquality.exceptions import GalileoException
from dataquality.integrations.hf import _validate_dataset, tokenize_adjust_labels
from dataquality.schemas.ner import TaggingSchema
from dataquality.utils.hf_tokenizer import (
    extract_gold_spans_at_word_level,
    infer_schema,
)
from tests.utils.hf_integration_constants import (
    ADJUSTED_TOKEN_DATA,
    UNADJUSTED_TOKEN_DATA,
    BIOSequence,
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


@pytest.mark.skip
def test_tokenize_adjust_labels() -> None:
    # TODO how do we test this without downloading from the internet
    # tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    # ds = load_dataset("wikiann", "it")
    tokenizer = mock.Mock()
    output = tokenize_adjust_labels(UNADJUSTED_TOKEN_DATA, tokenizer)
    for k in ADJUSTED_TOKEN_DATA:
        assert ADJUSTED_TOKEN_DATA[k] == output[k]


@pytest.mark.parametrize(
    "gold_sequence,gold_span",
    list(zip(BIOSequence.gold_sequences, BIOSequence.gold_spans)),
)
def test_extract_gold_spans_at_word_level_bio(
    gold_sequence: List[str], gold_span: List[Dict]
) -> None:
    # TODO: Our infer_schema function isnt good. If a gold sequence has {B, I}
    #  but no O, it throws an error because we need BIO or BIOES all/nothing.
    #  we need a more robust solution
    assert extract_gold_spans_at_word_level(gold_sequence) == gold_span


@pytest.mark.parametrize(
    "gold_sequence,gold_span",
    list(zip(BIOSequence.gold_sequences, BIOSequence.gold_spans)),
)
def test_extract_gold_spans_at_word_level_bioes(
    gold_sequence: List[str], gold_span: List[Dict]
) -> None:
    assert extract_gold_spans_at_word_level(gold_sequence) == gold_span


def test_extract_gold_spans_at_word_level_bilou() -> None:
    """TODO: Nidhi"""


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


@mock.patch("dataquality.integrations.hf.dataquality")
def test_tokenize_and_align_labels(mock_dq: mock.MagicMock) -> None:
    """TODO @nidhi

    Tests the e2e function call, passing in a DatasetDict and receiving a
    new DatasetDict, and that the datasets per split were logged correctly.
    """
