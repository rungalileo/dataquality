from typing import Callable
from unittest import mock

import datasets
import pandas as pd
import pytest
import vaex

import dataquality as dq
from dataquality.loggers.data_logger.base_data_logger import DataSet
from dataquality.loggers.data_logger.seq2seq import Seq2SeqDataLogger
from dataquality.schemas.split import Split
from tests.conftest import TestSessionVariables, tokenizer


@pytest.mark.parametrize(
    "dataset",
    [
        pd.DataFrame(
            {
                "summary": ["summary 1", "summary 2", "summary 3"],
                "title": ["title_1", "title_2", "title_3"],
                "my_id": [1, 2, 3],
            }
        ),
        vaex.from_dict(
            {
                "summary": ["summary 1", "summary 2", "summary 3"],
                "title": ["title_1", "title_2", "title_3"],
                "my_id": [1, 2, 3],
            }
        ),
        datasets.Dataset.from_dict(
            dict(
                summary=["summary 1", "summary 2", "summary 3"],
                title=["title_1", "title_2", "title_3"],
                my_id=[1, 2, 3],
            )
        ),
    ],
)
def test_log_dataset(
    dataset: DataSet,
    set_test_config: Callable,
    cleanup_after_use: Callable,
    test_session_vars: TestSessionVariables,
) -> None:
    set_test_config(task_type="seq2seq")
    logger = Seq2SeqDataLogger()

    with mock.patch("dataquality.core.log.get_data_logger") as mock_method:
        mock_method.return_value = logger
        dq.set_tokenizer(tokenizer)
        dq.log_dataset(
            dataset, text="summary", label="title", id="my_id", split="train"
        )

        assert logger.texts == ["summary 1", "summary 2", "summary 3"]
        assert logger.labels == ["title_1", "title_2", "title_3"]
        assert logger.ids == [1, 2, 3]
        assert logger.split == Split.training

    df = vaex.open(f"{test_session_vars.LOCATION}/input_data/training/data_0.arrow")
    expected_cols = [
        "id",
        "text",
        "label",
        "tokenized_label",
        "token_label_positions",
        "token_label_offsets",
    ]
    assert sorted(df.get_column_names()) == sorted(expected_cols)


def test_log_dataset_no_tokenizer(set_test_config: Callable) -> None:
    set_test_config(task_type="seq2seq")
    df = pd.DataFrame(
        {
            "summary": ["summary 1", "summary 2", "summary 3"],
            "title": ["title_1", "title_2", "title_3"],
            "my_id": [1, 2, 3],
        }
    )
    logger = Seq2SeqDataLogger()
    with mock.patch("dataquality.core.log.get_data_logger") as mock_method:
        mock_method.return_value = logger
        with pytest.raises(AssertionError) as e:
            dq.log_dataset(df, text="summary", label="title", id="my_id", split="train")
    assert str(e.value) == (
        "You must set your tokenizer before logging. Use `dq.set_tokenizer`"
    )
