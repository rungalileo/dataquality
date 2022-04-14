from typing import Callable
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import vaex

import dataquality as dq
from dataquality.exceptions import GalileoException
from dataquality.loggers.data_logger.base_data_logger import DataSet
from dataquality.loggers.data_logger.text_multi_label import TextMultiLabelDataLogger
from dataquality.schemas.split import Split


def test_duplicate_rows(set_test_config, cleanup_after_use) -> None:
    set_test_config(task_type="text_multi_label")
    text_inputs = [
        "what movies star bruce willis",
        "show me films with drew barrymore from the 1980s",
        "what movies starred both al pacino and robert deniro",
        "find me all of the movies that starred harold ramis and bill murray",
        "find me a movie with a quote about baseball in it",
    ]
    gold = [["A", "C", "B"]] * 5
    ids = [0, 1, 2, 3, 4]

    dq.log_data_samples(
        texts=text_inputs, task_labels=gold, split="validation", ids=ids
    )
    dq.log_data_samples(texts=text_inputs, task_labels=gold, split="training", ids=ids)

    with pytest.raises(GalileoException):
        dq.log_data_samples(
            texts=text_inputs, task_labels=gold, split="validation", ids=ids
        )

    dq.log_data_samples(texts=text_inputs, task_labels=gold, split="test", ids=ids)

    with pytest.raises(GalileoException):
        dq.log_data_samples(texts=text_inputs, task_labels=gold, split="test", ids=ids)


def test_duplicate_outputs_rows(set_test_config, cleanup_after_use) -> None:
    set_test_config(task_type="text_multi_label")
    num_samples = 5
    num_tasks = 4
    classes_per_task = 3
    text_inputs = [
        "what movies star bruce willis",
        "show me films with drew barrymore from the 1980s",
        "what movies starred both al pacino and robert deniro",
        "find me all of the movies that starred harold ramis and bill murray",
        "find me a movie with a quote about baseball in it",
    ]
    gold = [["A", "C", "B", "D"]] * 5
    ids = list(range(5))

    dq.log_data_samples(
        texts=text_inputs, task_labels=gold, split="validation", ids=ids
    )
    dq.log_data_samples(texts=text_inputs, task_labels=gold, split="training", ids=ids)

    embs = np.random.rand(num_samples, 100)
    logits = [[np.random.rand(classes_per_task)] * num_tasks] * num_samples
    ids = list(range(5))
    dq.log_model_outputs(embs=embs, logits=logits, ids=ids, split="training", epoch=0)
    dq.log_model_outputs(embs=embs, logits=logits, ids=ids, split="training", epoch=0)

    with pytest.raises(GalileoException) as e:
        dq.get_data_logger().upload()

    assert str(e.value).startswith("It seems as though you do not have unique ids")


def test_log_data_sample(
    set_test_config: Callable, cleanup_after_use: Callable
) -> None:
    set_test_config(task_type="text_multi_label")
    logger = TextMultiLabelDataLogger()
    with mock.patch("dataquality.core.log.get_data_logger") as mock_method:
        mock_method.return_value = logger
        dq.log_data_sample(text="sample 1", task_labels="A", id=1, split="training")

        assert logger.texts == ["sample 1"]
        assert logger.labels == ["A"]
        assert logger.ids == [1]
        assert logger.split == Split.training


@pytest.mark.parametrize(
    "dataset",
    [
        pd.DataFrame(
            {
                "my_text": ["sample1", "sample2", "sample3"],
                "my_labels": [["A", "A", "B"], ["C", "A", "B"], ["B", "C", "B"]],
                "my_id": [1, 2, 3],
            }
        ),
        vaex.from_pandas(
            pd.DataFrame(
                {
                    "my_text": ["sample1", "sample2", "sample3"],
                    "my_labels": [["A", "A", "B"], ["C", "A", "B"], ["B", "C", "B"]],
                    "my_id": [1, 2, 3],
                }
            )
        ),
        [
            {"my_text": "sample1", "my_labels": ["A", "A", "B"], "my_id": 1},
            {"my_text": "sample2", "my_labels": ["C", "A", "B"], "my_id": 2},
            {"my_text": "sample3", "my_labels": ["B", "C", "B"], "my_id": 3},
        ],
    ],
)
def test_log_dataset(
    dataset: DataSet, set_test_config: Callable, cleanup_after_use: Callable
) -> None:
    logger = TextMultiLabelDataLogger()

    with mock.patch("dataquality.core.log.get_data_logger") as mock_method:
        mock_method.return_value = logger
        dq.log_dataset(
            dataset, text="my_text", label="my_labels", id="my_id", split="train"
        )

        assert logger.texts == ["sample1", "sample2", "sample3"]
        log_labels = [list(i) for i in logger.labels]
        assert log_labels == [["A", "A", "B"], ["C", "A", "B"], ["B", "C", "B"]]
        assert logger.ids == [1, 2, 3]
        assert logger.split == Split.training


def test_log_dataset_tuple(
    set_test_config: Callable, cleanup_after_use: Callable
) -> None:
    logger = TextMultiLabelDataLogger()
    dataset = [
        ("sample1", ["A", "A", "B"], 1),
        ("sample2", ["C", "A", "B"], 2),
        ("sample3", ["B", "C", "B"], 3),
    ]

    with mock.patch("dataquality.core.log.get_data_logger") as mock_method:
        mock_method.return_value = logger
        dq.log_dataset(dataset, text=0, label=1, id=2, split="train")

        assert logger.texts == ["sample1", "sample2", "sample3"]
        log_labels = [list(i) for i in logger.labels]
        assert log_labels == [["A", "A", "B"], ["C", "A", "B"], ["B", "C", "B"]]
        assert logger.ids == [1, 2, 3]
        assert logger.split == Split.training
