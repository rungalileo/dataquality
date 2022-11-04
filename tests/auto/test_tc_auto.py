from tempfile import NamedTemporaryFile
from unittest import mock

import pandas as pd
import pytest
import vaex
from datasets import ClassLabel, Dataset, DatasetDict

from dataquality.auto.text_classification import (
    _add_class_label_to_dataset,
    _get_dataset,
    _process_pandas_df,
    _validate_dataset_dict,
)
from dataquality.exceptions import GalileoException
from dataquality.schemas.split import Split


def test_process_pandas_df() -> None:
    labels = ["red", "blue", "green"]
    df = pd.DataFrame(
        {"text": ["sample1", "sample2", "sample3"], "label": ["green", "blue", "green"]}
    )
    ds = _process_pandas_df(df, labels)
    assert isinstance(ds, Dataset)
    assert isinstance(ds.features["label"], ClassLabel)
    assert ds["label"] == [2, 1, 2]


def test_process_pandas_df_no_labels_provided() -> None:
    df = pd.DataFrame(
        {"text": ["sample1", "sample2", "sample3"], "label": ["green", "blue", "green"]}
    )
    ds = _process_pandas_df(df)
    assert isinstance(ds, Dataset)
    assert isinstance(ds.features["label"], ClassLabel)
    assert ds["label"] == [1, 0, 1]


def test_process_pandas_df_labels_are_ints() -> None:
    df = pd.DataFrame({"text": ["sample1", "sample2", "sample3"], "label": [0, 0, 1]})
    ds = _process_pandas_df(df)
    assert isinstance(ds, Dataset)
    # Cant crete class label because we dont have string labels available
    assert not isinstance(ds.features["label"], ClassLabel)
    assert ds["label"] == [0, 0, 1]


def test_process_pandas_df_labels_are_ints_labels_provided() -> None:
    df = pd.DataFrame({"text": ["sample1", "sample2", "sample3"], "label": [0, 0, 1]})
    labels = ["red", "green", "blue"]
    ds = _process_pandas_df(df, labels)
    assert isinstance(ds, Dataset)
    # Cant crete class label because we dont have string labels available
    assert isinstance(ds.features["label"], ClassLabel)
    assert ds.features["label"].names == labels
    assert ds["label"] == [0, 0, 1]


def test_process_pandas_df_labels_no_label_col() -> None:
    df = pd.DataFrame(
        {
            "text": ["sample1", "sample2", "sample3"],
        }
    )
    labels = ["red", "green", "blue"]
    ds = _process_pandas_df(df, labels)
    assert isinstance(ds, Dataset)
    assert "label" not in ds.features


def test_add_class_label_to_dataset() -> None:
    ds = Dataset.from_dict(
        {"text": ["sample1", "sample2", "sample3"], "label": ["green", "blue", "green"]}
    )
    ds = _add_class_label_to_dataset(ds)
    assert ds["label"] == [1, 0, 1]
    assert ds.features["label"].names == ["blue", "green"]


def test_add_class_label_to_dataset_int_fields() -> None:
    ds = Dataset.from_dict(
        {"text": ["sample1", "sample2", "sample3"], "label": [1, 0, 1]}
    )
    ds = _add_class_label_to_dataset(ds)
    assert ds["label"] == [1, 0, 1]
    assert isinstance(ds.features["label"], ClassLabel)
    assert ds.features["label"].names == [0, 1]


def test_add_class_label_to_dataset_no_label() -> None:
    ds = Dataset.from_dict(
        {
            "text": ["sample1", "sample2", "sample3"],
        }
    )
    ds = _add_class_label_to_dataset(ds)
    assert "label" not in ds.features


def test_add_class_label_to_dataset_int_labels_label_list_provided() -> None:
    ds = Dataset.from_dict(
        {"text": ["sample1", "sample2", "sample3"], "label": [1, 0, 1]}
    )
    labels = ["red", "green", "blue"]
    ds = _add_class_label_to_dataset(ds, labels)
    assert ds["label"] == [1, 0, 1]
    assert isinstance(ds.features["label"], ClassLabel)
    assert ds.features["label"].names == labels


def test_get_dataset() -> None:
    ds = Dataset.from_dict(
        {"text": ["sample1", "sample2", "sample3"], "label": [1, 0, 1]}
    )
    get_ds = _get_dataset(ds)
    assert isinstance(get_ds, Dataset)
    assert get_ds is ds


def test_get_dataset_from_pandas() -> None:
    labels = ["red", "blue", "green"]
    df = pd.DataFrame(
        {"text": ["sample1", "sample2", "sample3"], "label": ["green", "blue", "green"]}
    )
    ds = _get_dataset(df, labels)
    assert isinstance(ds, Dataset)
    assert ds["label"] == [2, 1, 2]


def test_get_dataset_from_file() -> None:
    labels = ["red", "blue", "green"]
    df = pd.DataFrame(
        {"text": ["sample1", "sample2", "sample3"], "label": ["green", "blue", "green"]}
    )
    ds = _get_dataset(df, labels)
    assert isinstance(ds, Dataset)
    assert ds["label"] == [2, 1, 2]
    with NamedTemporaryFile(suffix=".csv") as f:
        df.to_csv(f.name)
        ds = _get_dataset(f.name, labels)
        assert isinstance(ds, Dataset)
        assert ds["label"] == [2, 1, 2]


def test_get_dataset_from_vaex() -> None:
    df = pd.DataFrame(
        {"text": ["sample1", "sample2", "sample3"], "label": ["green", "blue", "green"]}
    )
    df = vaex.from_pandas(df)
    with pytest.raises(GalileoException) as e:
        _get_dataset(df)
    assert str(e.value) == (
        "Dataset must be one of pandas df, huggingface Dataset, or string path"
    )


@mock.patch("dataquality.auto.text_classification.load_dataset")
def test_get_dataset_from_huggingface(mock_load_dataset: mock.MagicMock) -> None:
    ds = Dataset.from_dict(
        {"text": ["sample1", "sample2", "sample3"], "label": [1, 0, 1]}
    )
    mock_load_dataset.return_value = ds
    get_ds = _get_dataset("huggingface/path")
    assert isinstance(get_ds, Dataset)
    assert get_ds is ds


@mock.patch("dataquality.auto.text_classification.load_dataset")
def test_get_dataset_from_huggingface_not_dataset(
    mock_load_dataset: mock.MagicMock,
) -> None:
    ds = Dataset.from_dict(
        {"text": ["sample1", "sample2", "sample3"], "label": [1, 0, 1]}
    )
    dd = DatasetDict({"train": ds})
    # It can't return this. Must be a dataset in this function
    mock_load_dataset.return_value = dd
    with pytest.raises(AssertionError) as e:
        _get_dataset("huggingface_path")
    assert str(e.value).startswith("Loaded data should be of type Dataset, but found")


def test_validate_dataset_dict() -> None:
    ds = Dataset.from_dict(
        {"text": ["sample1", "sample2", "sample3"], "label": [1, 0, 1]}
    )
    dd = DatasetDict({"train": ds})
    new_dd = _validate_dataset_dict(dd)
    assert new_dd[Split.train]["id"] == [0, 1, 2]


def test_validate_dataset_dict_no_labels() -> None:
    ds = Dataset.from_dict(
        {
            "text": ["sample1", "sample2", "sample3"],
        }
    )
    dd = DatasetDict({"train": ds})
    with pytest.raises(AssertionError) as e:
        _validate_dataset_dict(dd)
    assert str(e.value) == "Dataset must have column `label`"


def test_validate_dataset_dict_no_text() -> None:
    ds = Dataset.from_dict({"label": [1, 0, 1]})
    dd = DatasetDict({"train": ds})
    with pytest.raises(AssertionError) as e:
        _validate_dataset_dict(dd)
    assert str(e.value) == "Dataset must have column `text`"
