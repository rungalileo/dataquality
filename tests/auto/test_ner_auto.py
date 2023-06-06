from typing import Callable, Union
from unittest import mock

import pandas as pd
import pytest
from datasets import Dataset, DatasetDict

import dataquality as dq
from dataquality.dq_auto.ner import NERDatasetManager, auto
from dataquality.schemas.split import Split

base_df = df = pd.DataFrame(
    {"tokens": ["the", "thing", "is", "blue"], "tags": [0, 1, 1, 2]}
)
manager = NERDatasetManager()


@pytest.mark.parametrize("data", [Dataset.from_pandas(base_df), base_df, "testdata"])
@mock.patch(
    "dataquality.dq_auto.base_data_manager.load_data_from_str", return_value=base_df
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
    valid_dd = manager._validate_dataset_dict(dd, inference_names=[])
    assert Split.train in valid_dd and Split.validation in valid_dd


@mock.patch("dataquality.finish")
@mock.patch("dataquality.utils.auto_trainer.watch")
@mock.patch("dataquality.dq_auto.ner.get_trainer")
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
    assert dd[Split.train].num_rows == 3
    assert dd[Split.validation].num_rows == 1


@mock.patch("dataquality.finish")
@mock.patch("dataquality.utils.auto_trainer.watch")
@mock.patch("dataquality.dq_auto.ner.get_trainer")
@mock.patch("dataquality.log_dataset")
@mock.patch("dataquality.set_labels_for_run")
@mock.patch("dataquality.init")
@mock.patch("dataquality.login")
def test_call_auto_pandas_with_inference(
    mock_login: mock.MagicMock,
    mock_init: mock.MagicMock,
    mock_set_labels: mock.MagicMock,
    mock_log_dataset: mock.MagicMock,
    mock_get_trainer: mock.MagicMock,
    mock_watch: mock.MagicMock,
    mock_finish: mock.MagicMock,
    set_test_config: Callable,
) -> None:
    df_train = base_df.copy()
    df_inf = pd.DataFrame({"tokens": ["some", "new", "tokens", "heree"]})
    trainer = mock.MagicMock()
    encoded_data = {Split.training: df_train, "inf_1": df_inf}
    mock_get_trainer.return_value = trainer, encoded_data
    auto(train_data=df_train, inference_data={"inf_1": df_inf})
    trainer.predict.assert_called_once()
    assert dq.get_data_logger().logger_config.cur_split == Split.inference
    assert dq.get_data_logger().logger_config.cur_inference_name == "inf_1"

    mock_get_trainer.assert_called_once()
    dd = mock_get_trainer.call_args_list[0][0][0]
    assert Split.train in dd and Split.validation in dd
    assert dd[Split.train].num_rows == 3
    assert dd[Split.validation].num_rows == 1
    assert dd["inf_1"].num_rows == 4
