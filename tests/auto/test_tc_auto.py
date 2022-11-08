from tempfile import NamedTemporaryFile
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import vaex
from datasets import ClassLabel, Dataset, DatasetDict

from dataquality.auto.text_classification import (
    DEMO_DATASETS,
    _add_class_label_to_dataset,
    _convert_df_to_dataset,
    _convert_to_hf_dataset,
    _get_dataset_dict,
    _get_labels,
    _log_dataset_dict,
    _validate_dataset_dict,
    auto,
)
from dataquality.exceptions import GalileoException
from dataquality.schemas.split import Split


def test_convert_df_to_dataset() -> None:
    labels = ["red", "blue", "green"]
    df = pd.DataFrame(
        {"text": ["sample1", "sample2", "sample3"], "label": ["green", "blue", "green"]}
    )
    ds = _convert_df_to_dataset(df, labels)
    assert isinstance(ds, Dataset)
    assert isinstance(ds.features["label"], ClassLabel)
    assert ds["label"] == [2, 1, 2]


def test_convert_df_to_dataset_no_labels_provided() -> None:
    df = pd.DataFrame(
        {"text": ["sample1", "sample2", "sample3"], "label": ["green", "blue", "green"]}
    )
    ds = _convert_df_to_dataset(df)
    assert isinstance(ds, Dataset)
    assert isinstance(ds.features["label"], ClassLabel)
    assert ds["label"] == [1, 0, 1]


def test_convert_df_to_dataset_labels_are_ints() -> None:
    df = pd.DataFrame({"text": ["sample1", "sample2", "sample3"], "label": [0, 0, 1]})
    ds = _convert_df_to_dataset(df)
    assert isinstance(ds, Dataset)
    assert isinstance(ds.features["label"], ClassLabel)
    assert ds["label"] == [0, 0, 1]


def test_convert_df_to_dataset_labels_are_ints_labels_provided() -> None:
    df = pd.DataFrame({"text": ["sample1", "sample2", "sample3"], "label": [0, 0, 1]})
    labels = ["red", "green", "blue"]
    ds = _convert_df_to_dataset(df, labels)
    assert isinstance(ds, Dataset)
    # Cant crete class label because we dont have string labels available
    assert isinstance(ds.features["label"], ClassLabel)
    assert ds.features["label"].names == labels
    assert ds["label"] == [0, 0, 1]


def test_convert_df_to_dataset_labels_no_label_col() -> None:
    df = pd.DataFrame(
        {
            "text": ["sample1", "sample2", "sample3"],
        }
    )
    labels = ["red", "green", "blue"]
    ds = _convert_df_to_dataset(df, labels)
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
    get_ds = _convert_to_hf_dataset(ds)
    assert isinstance(get_ds, Dataset)
    assert get_ds is ds


def test_get_dataset_from_pandas() -> None:
    labels = ["red", "blue", "green"]
    df = pd.DataFrame(
        {"text": ["sample1", "sample2", "sample3"], "label": ["green", "blue", "green"]}
    )
    ds = _convert_to_hf_dataset(df, labels)
    assert isinstance(ds, Dataset)
    assert ds["label"] == [2, 1, 2]


def test_get_dataset_from_file() -> None:
    labels = ["red", "blue", "green"]
    df = pd.DataFrame(
        {"text": ["sample1", "sample2", "sample3"], "label": ["green", "blue", "green"]}
    )
    ds = _convert_to_hf_dataset(df, labels)
    assert isinstance(ds, Dataset)
    assert ds["label"] == [2, 1, 2]
    with NamedTemporaryFile(suffix=".csv") as f:
        df.to_csv(f.name)
        ds = _convert_to_hf_dataset(f.name, labels)
        assert isinstance(ds, Dataset)
        assert ds["label"] == [2, 1, 2]


def test_get_dataset_from_vaex() -> None:
    df = pd.DataFrame(
        {"text": ["sample1", "sample2", "sample3"], "label": ["green", "blue", "green"]}
    )
    df = vaex.from_pandas(df)
    with pytest.raises(GalileoException) as e:
        _convert_to_hf_dataset(df)
    assert str(e.value) == (
        "Dataset must be one of pandas df, huggingface Dataset, or string path"
    )


@mock.patch("dataquality.auto.text_classification.load_dataset")
def test_get_dataset_from_huggingface(mock_load_dataset: mock.MagicMock) -> None:
    ds = Dataset.from_dict(
        {"text": ["sample1", "sample2", "sample3"], "label": [1, 0, 1]}
    )
    mock_load_dataset.return_value = ds
    get_ds = _convert_to_hf_dataset("huggingface/path")
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
        _convert_to_hf_dataset("huggingface_path")
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


@mock.patch("dataquality.auto.text_classification.load_dataset")
def test_get_dataset_dict_no_dataset(mock_load_dataset: mock.MagicMock) -> None:
    dd = DatasetDict(
        {
            "train": Dataset.from_dict(
                {"text": ["sample1", "sample2", "sample3"], "label": [1, 0, 1]}
            )
        }
    )
    mock_load_dataset.return_value = dd
    get_dd = _get_dataset_dict()
    assert isinstance(get_dd, DatasetDict)
    assert get_dd is dd
    assert mock_load_dataset.call_args_list[0][0][0] in DEMO_DATASETS


@mock.patch("dataquality.auto.text_classification.load_dataset")
def test_get_dataset_dict_not_dataset_dict(mock_load_dataset: mock.MagicMock) -> None:
    ds = Dataset.from_dict(
        {"text": ["sample1", "sample2", "sample3"], "label": [1, 0, 1]}
    )
    mock_load_dataset.return_value = ds
    with pytest.raises(AssertionError) as e:
        _get_dataset_dict()
    assert str(e.value).startswith(
        "hf_data must be a path to a huggingface DatasetDict"
    )


def test_get_dataset_dict() -> None:
    dd = DatasetDict(
        {
            "train": Dataset.from_dict(
                {"text": ["sample1", "sample2", "sample3"], "label": [1, 0, 1]}
            )
        }
    )
    assert _get_dataset_dict(hf_data=dd) is dd


def test_get_dataset_dict_no_hf_data() -> None:
    ds_train = Dataset.from_dict(
        {"text": ["sample1", "sample2", "sample3"], "label": [1, 0, 1]}
    )
    ds_test = Dataset.from_dict(
        {"text": ["sample4", "sample5", "sample6"], "label": [0, 0, 0]}
    )
    ds_val = Dataset.from_dict(
        {"text": ["sample7", "sample8", "sample9"], "label": [1, 1, 0]}
    )
    dd = _get_dataset_dict(train_data=ds_train, val_data=ds_val, test_data=ds_test)
    for key in dd:
        assert key in list(Split)
    assert dd[Split.train]["text"] == ds_train["text"]
    assert dd[Split.validation]["text"] == ds_val["text"]
    assert dd[Split.test]["text"] == ds_test["text"]


@pytest.mark.parametrize("as_numpy", [True, False])
def test_get_labels(as_numpy: bool) -> None:
    dd = DatasetDict(
        {
            "train": Dataset.from_dict(
                {"text": ["sample1", "sample2", "sample3"], "label": [1, 0, 1]}
            )
        }
    )
    labels = np.array(["a", "b", "c"]) if as_numpy else ["a", "b", "c"]
    assert _get_labels(dd, labels) == list(labels)


def test_get_labels_no_labels() -> None:
    dd = DatasetDict(
        {
            Split.train: Dataset.from_dict(
                {"text": ["sample1", "sample2", "sample3"], "label": [1, 0, 1]}
            )
        }
    )
    assert _get_labels(dd) == [0, 1]


def test_get_labels_class_label() -> None:
    dd = DatasetDict(
        {
            Split.train: Dataset.from_dict(
                {"text": ["sample1", "sample2", "sample3"], "label": [1, 0, 1]}
            )
        }
    )
    class_label = ClassLabel(num_classes=3, names=["red", "green", "blue"])
    dd[Split.train] = dd[Split.train].cast_column("label", class_label)
    dd[Split.train] = dd[Split.train]
    assert _get_labels(dd) == ["red", "green", "blue"]


@mock.patch("dataquality.log_dataset")
def test_log_dataset_dict(mock_log_ds: mock.MagicMock) -> None:
    ds_train = Dataset.from_dict(
        {
            "text": ["sample1", "sample2", "sample3"],
            "label": [1, 0, 1],
            "id": [0, 1, 2],
            "meta_0": ["cat", "dog", "55"],
        }
    )
    ds_test = Dataset.from_dict(
        {"text": ["sample4", "sample5", "sample6"], "label": [0, 0, 0], "id": [0, 1, 2]}
    )
    ds_val = Dataset.from_dict(
        {
            "text": ["sample7", "sample8", "sample9"],
            "label": [1, 1, 0],
            "id": [0, 1, 2],
            "my_meta": [0.55, -1, 33],
        }
    )
    dd = DatasetDict(
        {
            Split.train: ds_train,
            Split.test: ds_test,
            Split.validation: ds_val,
        }
    )
    _log_dataset_dict(dd)
    assert mock_log_ds.call_count == 3
    for arg_list in mock_log_ds.call_args_list:
        kwargs = arg_list[1]
        if kwargs["split"] == Split.train:
            assert kwargs["meta"] == ["meta_0"]
        if kwargs["split"] == Split.test:
            assert kwargs["meta"] == []
        if kwargs["split"] == Split.validation:
            assert kwargs["meta"] == ["my_meta"]


@pytest.mark.parametrize("use_ids", [True, False])
@mock.patch("dataquality.finish")
@mock.patch("dataquality.auto.text_classification.watch")
@mock.patch("dataquality.auto.text_classification.get_trainer")
@mock.patch("dataquality.log_dataset")
@mock.patch("dataquality.set_labels_for_run")
@mock.patch("dataquality.init")
@mock.patch("dataquality.login")
def test_call_auto_pandas_train_df_mixed_meta(
    mock_login: mock.MagicMock,
    mock_init: mock.MagicMock,
    mock_set_labels: mock.MagicMock,
    mock_log_dataset: mock.MagicMock,
    mock_get_trainer: mock.MagicMock,
    mock_watch: mock.MagicMock,
    mock_finish: mock.MagicMock,
    use_ids: bool,
) -> None:
    df_train = pd.DataFrame(
        {
            "text": ["sample4", "sample5", "sample6"],
            "label": ["red", "green", "blue"],
            "meta_1": [0, "apple", 5.42],
        }
    )
    if use_ids:
        df_train["id"] = list(range(len(df_train)))
    trainer = mock.MagicMock()
    encoded_data = {}
    mock_get_trainer.return_value = trainer, encoded_data
    auto(train_data=df_train)

    mock_log_dataset.assert_called_once()
    args, kwargs = mock_log_dataset.call_args_list[0]
    assert kwargs["meta"] == ["meta_1"]
    # Whether or not we provided the index IDs, they should be added and logged
    ds_logged_arg = args[0]
    assert ds_logged_arg["id"] == [0, 1, 2]
