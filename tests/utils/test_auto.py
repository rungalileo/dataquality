from unittest import mock

import pandas as pd
import pytest
from datasets import Dataset, DatasetDict

from dataquality.exceptions import GalileoException
from dataquality.schemas.split import Split
from dataquality.utils.auto import (
    add_val_data_if_missing,
    load_data_from_str,
    open_console_url,
)


def test_add_val_data_if_missing() -> None:
    df = pd.DataFrame(
        {"text": ["sample1", "sample2", "sample3"], "label": ["green", "blue", "green"]}
    )
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


@mock.patch("dataquality.utils.auto.webbrowser")
def test_open_console_raises_exc(mock_browser: mock.MagicMock):
    """Should catch the exception silently"""
    mock_open = mock.MagicMock()
    mock_open.side_effect = Exception("Bad!")
    mock_browser.open = mock_open
    open_console_url("https://console.cloud.rungalileo.io")
    mock_open.assert_called_once()
