from typing import Any
from unittest import mock

import pandas as pd
import pytest
from datasets import Dataset, DatasetDict

from dataquality.exceptions import GalileoException
from dataquality.schemas.split import Split
from dataquality.schemas.task_type import TaskType
from dataquality.utils.auto import (
    _get_task_type_from_cols,
    add_val_data_if_missing,
    get_task_type_from_data,
    load_data_from_str,
    run_name_from_hf_dataset,
)
from dataquality.utils.helpers import open_console_url

TC_DATA = pd.DataFrame(
    {"text": ["sample1", "sample2", "sample3"], "label": ["green", "blue", "green"]}
)
NER_DATA = pd.DataFrame(
    {"tokens": ["the", "thing", "is", "blue"], "tags": [0, 1, 1, 2]}
)
NER_DATA2 = pd.DataFrame(
    {"tokens": ["the", "thing", "is", "blue"], "ner_tags": [0, 1, 1, 2]}
)


def test_add_val_data_if_missing() -> None:
    df = TC_DATA.copy()
    dd = DatasetDict({Split.train: Dataset.from_pandas(df)})
    split_dd = add_val_data_if_missing(dd)
    assert Split.train in split_dd and Split.validation in split_dd
    assert len(split_dd[Split.train]) == 2
    assert len(split_dd[Split.validation]) == 1


def test_add_val_data_if_missing_has_test() -> None:
    df = pd.DataFrame(
        {"text": ["sample1", "sample2", "sample3"], "label": ["green", "blue", "green"]}
    )
    dd = DatasetDict(
        {
            Split.train: Dataset.from_pandas(df),
            Split.test: Dataset.from_pandas(df),
        }
    )
    test_ds = dd[Split.test]
    split_dd = add_val_data_if_missing(dd)
    assert Split.train in split_dd and Split.validation in split_dd
    assert Split.test not in split_dd
    assert sorted(split_dd[Split.validation]["text"]) == sorted(test_ds["text"])
    assert sorted(split_dd[Split.validation]["label"]) == sorted(test_ds["label"])


def test_add_val_data_if_missing_has_val() -> None:
    df = pd.DataFrame(
        {"text": ["sample1", "sample2", "sample3"], "label": ["green", "blue", "green"]}
    )
    dd = DatasetDict(
        {
            Split.train: Dataset.from_pandas(df),
            Split.validation: Dataset.from_pandas(df),
        }
    )
    split_dd = add_val_data_if_missing(dd)
    assert dd is split_dd


def test_load_data_from_str_bad_extension() -> None:
    with pytest.raises(GalileoException) as e:
        load_data_from_str("file.invalidext")
    assert str(e.value).startswith(
        "Local file path extension must be readable by pandas"
    )


def test_open_console_url_no_url():
    """Shouldn't fail, shouldn't complain. Just run silently"""
    open_console_url()


@mock.patch("dataquality.utils.helpers.webbrowser")
def test_open_console_raises_exc(mock_browser: mock.MagicMock):
    """Should catch the exception silently"""
    mock_open = mock.MagicMock()
    mock_open.side_effect = Exception("Bad!")
    mock_browser.open = mock_open
    open_console_url("https://console.cloud.rungalileo.io")
    mock_open.assert_called_once()


@pytest.mark.parametrize(
    "hf_data,name",
    [
        ["rungalileo/my-data.set 1", "rungalileo_my-data_set 1"],
        ["connll 203", "connll 203"],
        ["data 1 f1 0.23", "data 1 f1 0_23"],
    ],
)
def test_run_name_from_hf_dataset(hf_data: str, name: str) -> None:
    assert run_name_from_hf_dataset(hf_data).startswith(name)


@pytest.mark.parametrize(
    "data,task_type",
    [
        [TC_DATA, TaskType.text_classification],
        [Dataset.from_pandas(TC_DATA), TaskType.text_classification],
        [NER_DATA, TaskType.text_ner],
        [Dataset.from_pandas(NER_DATA2), TaskType.text_ner],
    ],
)
def test_get_task_type_from_data(data: Any, task_type: TaskType) -> None:
    assert get_task_type_from_data(train_data=data) == task_type


def test_get_task_type_from_cols_invalid() -> None:
    with pytest.raises(GalileoException):
        _get_task_type_from_cols(["text", "tags"])


@pytest.mark.parametrize(
    "data,task_type",
    [
        [
            DatasetDict({"train": Dataset.from_pandas(TC_DATA)}),
            TaskType.text_classification,
        ],
        [DatasetDict({"train": Dataset.from_pandas(NER_DATA)}), TaskType.text_ner],
    ],
)
def test_get_task_type_from_data_hf_data(data: Any, task_type: TaskType) -> None:
    assert get_task_type_from_data(hf_data=data) == task_type


@mock.patch("dataquality.utils.auto.load_dataset")
def test_get_task_type_from_data_hf_data_wrong_type(
    mock_load_ds: mock.MagicMock,
) -> None:
    dd = Dataset.from_pandas(TC_DATA)
    mock_load_ds.return_value = dd
    with pytest.raises(AssertionError):
        get_task_type_from_data(hf_data="dataset_in_hf")
