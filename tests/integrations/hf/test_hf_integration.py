import json
from typing import Callable, Dict, List
from unittest import mock

import datasets
import numpy as np
import pytest

from dataquality.exceptions import GalileoException
from dataquality.integrations.hf import (
    _extract_labels_from_ds,
    _validate_dataset,
    get_dataloader,
    infer_schema,
    tokenize_adjust_labels,
    tokenize_and_log_dataset,
)
from dataquality.schemas.ner import TaggingSchema
from dataquality.schemas.split import Split
from dataquality.utils.hf_tokenizer import extract_gold_spans_at_word_level
from tests.test_utils.hf_integration_constants import (
    ADJUSTED_TOKEN_DATA,
    UNADJUSTED_TOKEN_DATA,
    BILOUSequence,
    BIOESSequence,
    BIOSequence,
    mock_ds,
    mock_tokenizer,
    tag_names,
)
from tests.test_utils.hf_integration_constants_inference import (
    ADJUSTED_TOKEN_DATA_INF,
    label_names,
    mock_ds_inf,
    mock_tokenizer_inf,
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
def test_tokenize_and_log_dataset(
    mock_log_dataset: mock.MagicMock, set_test_config
) -> None:
    """Tests the e2e function call, passing in a DatasetDict and receiving a

    new DatasetDict, and that the datasets per split were logged correctly.
    """
    set_test_config(task_type="text_ner")
    tokenize_output = tokenize_adjust_labels(mock_ds, mock_tokenizer, tag_names)
    with mock.patch("dataquality.integrations.hf.tokenize_adjust_labels") as mock_tok:
        mock_tok.return_value = tokenize_output
        ds_dict = datasets.DatasetDict(
            {
                "train": mock_ds,
                "test": mock_ds,
                "validation": mock_ds,
            }
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
    for split in [Split.train, Split.test, Split.validation]:
        mock_log_dataset.assert_any_call(mock.ANY, split=split, meta=[])


@mock.patch("dataquality.log_dataset")
def test_tokenize_and_log_dataset_inference(
    mock_log_dataset: mock.MagicMock, set_test_config
) -> None:
    """Tests the e2e function call, passing in a DatasetDict and receiving a

    new DatasetDict, and that the datasets per split were logged correctly.
    """
    set_test_config(task_type="text_ner")
    tokenize_output = tokenize_adjust_labels(mock_ds, mock_tokenizer, tag_names)
    tokenize_output_inf = tokenize_adjust_labels(
        mock_ds_inf, mock_tokenizer_inf, label_names
    )
    with mock.patch("dataquality.integrations.hf.tokenize_adjust_labels") as mock_tok:
        mock_tok.side_effect = [tokenize_output, tokenize_output_inf]
        ds_dict = datasets.DatasetDict(
            {
                "train": mock_ds,
                "inf1": mock_ds_inf,
            }
        )
        output = tokenize_and_log_dataset(ds_dict, mock_tokenizer, label_names)

    for split in ds_dict.keys():
        expected_data = (
            ADJUSTED_TOKEN_DATA_INF if split == "inf1" else ADJUSTED_TOKEN_DATA
        )
        split_output = output[split]
        for k in expected_data:
            token_data = expected_data[k]
            if k == "text_token_indices":
                # We abuse token data because outputs are returning tuples but we want
                # to compare lists
                token_data = json.loads(json.dumps(token_data))
            assert token_data == split_output[k]

    assert mock_log_dataset.call_count == 2
    mock_log_dataset.assert_any_call(mock.ANY, split=Split.training, meta=[])
    mock_log_dataset.assert_any_call(
        mock.ANY, split=Split.inference, meta=[], inference_name="inf1"
    )


def test_get_dataloader() -> None:
    dataset = ADJUSTED_TOKEN_DATA.copy()
    dataset["id"] = list(range(len(dataset["input_ids"])))
    ds = datasets.Dataset.from_dict(dataset)
    loader = get_dataloader(ds)
    assert len(loader.dataset) == len(mock_ds)
    assert sorted(list(loader.dataset[2].keys())) == sorted(
        ["id", "input_ids", "attention_mask", "labels"]
    )


def test_validate_dataset_no_tags() -> None:
    dataset = UNADJUSTED_TOKEN_DATA.copy()
    dataset.pop("ner_tags")
    ds = datasets.Dataset.from_dict(dataset)
    dd = datasets.DatasetDict({"train": ds})
    with pytest.raises(GalileoException) as e:
        _validate_dataset(dd)

    assert str(e.value) == "Each dataset must have either ner_tags or tags"


def test_validate_dataset_converts_tags_to_ner_tags() -> None:
    dataset = UNADJUSTED_TOKEN_DATA.copy()
    dataset["tags"] = dataset.pop("ner_tags")
    ds = datasets.Dataset.from_dict(dataset)
    dd = datasets.DatasetDict({"train": ds})
    dd = _validate_dataset(dd)
    assert "tags" not in dd["train"].features
    assert "ner_tags" in dd["train"].features


def test_extract_labels_from_ds_ner_labels() -> None:
    dataset = UNADJUSTED_TOKEN_DATA.copy()
    dataset["ner_labels"] = [["A", "B", "C"] for _ in range(len(dataset["ner_tags"]))]
    ds = datasets.Dataset.from_dict(dataset)
    dd = datasets.DatasetDict({"train": ds})
    assert _extract_labels_from_ds(dd) == ["A", "B", "C"]


def test_extract_labels_from_ds_no_labels() -> None:
    dataset = ADJUSTED_TOKEN_DATA.copy()
    ds = datasets.Dataset.from_dict(dataset)
    dd = datasets.DatasetDict({"train": ds})
    with pytest.raises(GalileoException) as e:
        _extract_labels_from_ds(dd)
    assert str(e.value).startswith("Could not extract labels from Dataset.")


def test_tokenize_and_log_dataset_invalid_labels() -> None:
    dd = datasets.DatasetDict({"train": mock_ds})
    with pytest.raises(AssertionError) as e:
        tokenize_and_log_dataset(dd, mock_tokenizer, np.array(["a", "b", "c"]))
    assert str(e.value).startswith("label_names must be of type list, but got")


@mock.patch("dataquality.log_dataset")
def test_tokenize_and_log_dataset_with_meta(
    mock_log_dataset: mock.MagicMock, set_test_config
) -> None:
    """Tests that with meta columns, they will be logged"""
    set_test_config(task_type="text_ner")
    tokenize_output = tokenize_adjust_labels(mock_ds, mock_tokenizer, tag_names)

    mock_ds_meta = mock_ds.add_column("test_meta_1", ["a", "b", "c", "d", "e"])
    with mock.patch("dataquality.integrations.hf.tokenize_adjust_labels") as mock_tok:
        mock_tok.return_value = tokenize_output
        # Only test and val have meta, make sure they both get logged with meta
        ds_dict = datasets.DatasetDict(
            {"train": mock_ds, "test": mock_ds_meta, "validation": mock_ds_meta}
        )
        output = tokenize_and_log_dataset(ds_dict, mock_tokenizer, meta=["test_meta_1"])
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
    call_args = mock_log_dataset.call_args_list

    # Training has no meta
    assert call_args[0][-1]["meta"] == []
    # Test and val do have meta
    assert call_args[1][-1]["meta"] == ["test_meta_1"]
    assert call_args[2][-1]["meta"] == ["test_meta_1"]


def test_validate_dataset_filters_empty_tokens(set_test_config: Callable) -> None:
    set_test_config(task_type="text_ner")
    empty_example = {
        "tokens": [["Swansea", "1", "Lincoln", "2"], []],
        "ner_tags": [[3, 0, 3, 0], []],
    }
    dd = datasets.DatasetDict({"train": datasets.Dataset.from_dict(empty_example)})
    assert dd["train"].num_rows == 2
    dd = _validate_dataset(dd)
    assert dd["train"].num_rows == 1
