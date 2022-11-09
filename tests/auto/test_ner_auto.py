from typing import Union
from unittest import mock

import pandas as pd
import pytest
from datasets import Dataset, DatasetDict

from dataquality.auto.ner import NERDatasetManager, auto
from dataquality.exceptions import GalileoException
from dataquality.schemas.split import Split

base_df = df = pd.DataFrame(
    {"tokens": ["the", "thing", "is", "blue"], "tags": [0, 1, 1, 2]}
)
manager = NERDatasetManager()


@pytest.mark.parametrize("key", ["tags", "ner_tags"])
def test_convert_to_hf_format(key: str) -> None:
    df[key] = base_df.copy().pop("tags")
    ds = Dataset.from_pandas(df)
    ds_convert = manager._convert_dataset_to_hf_format(ds)
    assert ds_convert == ds


def test_convert_to_hf_format_bad_cols() -> None:
    df = base_df[["tokens"]]
    ds = Dataset.from_pandas(df)
    with pytest.raises(GalileoException) as e:
        manager._convert_dataset_to_hf_format(ds)
    assert str(e.value).startswith("Data must be in either huggingface format")


@pytest.mark.parametrize("data", [Dataset.from_pandas(base_df), base_df, "testdata"])
@mock.patch(
    "dataquality.auto.base_data_manager.load_data_from_str", return_value=base_df
)
def test_convert_to_hf_dataset(
    mock_load: mock.MagicMock, data: Union[pd.DataFrame, Dataset, str]
) -> None:
    ds = manager._convert_to_hf_dataset(data)
    assert isinstance(ds, Dataset)
    assert "tokens" in ds.features
    assert "tags" in ds.features


def test_validate_dataset_dict() -> None:
    dd = DatasetDict({"train": Dataset.from_pandas(base_df)})
    valid_dd = manager._validate_dataset_dict(dd)
    assert Split.train in valid_dd and Split.validation in valid_dd


@mock.patch("dataquality.finish")
@mock.patch("dataquality.utils.auto.watch")
@mock.patch("dataquality.auto.ner.get_trainer")
@mock.patch("dataquality.log_dataset")
@mock.patch("dataquality.set_labels_for_run")
@mock.patch("dataquality.init")
@mock.patch("dataquality.login")
def test_call_auto_pandas_train_df(
    mock_login: mock.MagicMock,
    mock_init: mock.MagicMock,
    mock_set_labels: mock.MagicMock,
    mock_log_dataset: mock.MagicMock,
    mock_get_trainer: mock.MagicMock,
    mock_watch: mock.MagicMock,
    mock_finish: mock.MagicMock,
) -> None:

    df_train = base_df.copy()
    trainer = mock.MagicMock()
    encoded_data = {}
    mock_get_trainer.return_value = trainer, encoded_data
    auto(train_data=df_train)

    mock_get_trainer.assert_called_once()
    dd = mock_get_trainer.call_args_list[0][0][0]
    assert Split.train in dd and Split.validation in dd
    print("HERE")
    print(dd[Split.train])
    print(dd[Split.validation])
    assert dd[Split.train].num_rows == 3
    assert dd[Split.validation].num_rows == 1
