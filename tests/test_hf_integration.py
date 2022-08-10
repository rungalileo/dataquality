from typing import Dict, List
from unittest import mock

import dataquality.utils.hf_tokenizer
import datasets
import pytest
from unittest import mock

from dataquality.exceptions import GalileoException
from dataquality.integrations.hf import (
    _validate_dataset,
    infer_schema,
    tokenize_adjust_labels,
)
from dataquality.schemas.ner import TaggingSchema
from dataquality.utils.hf_tokenizer import extract_gold_spans_at_word_level
from tests.utils.hf_integration_constants import (
    ADJUSTED_TOKEN_DATA,
    UNADJUSTED_TOKEN_DATA,
    BIOESequence,
    BILOUSequence,
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
    tokenizer.batch_encode_plus.return_value = {'input_ids': [[101, 1005, 1005, 1005, 4080, 7015, 1005, 1005, 1005, 1011, 27424, 11261, 28101, 11639, 11261, 102], [101, 12005, 22311, 3406, 2632, 1018, 2102, 4830, 5557, 6264, 1031, 1017, 1033, 102], [101, 1005, 1005, 1005, 14278, 1005, 1005, 1005, 102], [101, 6148, 4313, 8661, 2572, 23550, 19763, 102], [101, 5292, 14163, 26302, 3406, 6335, 2053, 4168, 17488, 6178, 4747, 19098, 3995, 16985, 7011, 3148, 1012, 102]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]}
    output = tokenize_adjust_labels(UNADJUSTED_TOKEN_DATA, tokenizer)
    for k in ADJUSTED_TOKEN_DATA:
        assert ADJUSTED_TOKEN_DATA[k] == output[k]


@pytest.mark.parametrize(
    "gold_sequence,gold_span",
    list(zip(BIOESequence.gold_sequences, BIOESequence.gold_spans)),
)
def test_extract_gold_spans_at_word_level_bio(
    gold_sequence: List[str], gold_span: List[Dict]
) -> None:
    assert (
        extract_gold_spans_at_word_level(gold_sequence, TaggingSchema.BIO) == gold_span
    )


@pytest.mark.parametrize(
    "gold_sequence,gold_span",
    list(zip(BIOESequence.gold_sequences, BIOESequence.gold_spans)),
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


@mock.patch("dataquality.integrations.hf.dataquality")
def test_tokenize_and_and_log_dataset(mock_dq: mock.MagicMock) -> None:
    """TODO @nidhi

    Tests the e2e function call, passing in a DatasetDict and receiving a
    new DatasetDict, and that the datasets per split were logged correctly.
    """
