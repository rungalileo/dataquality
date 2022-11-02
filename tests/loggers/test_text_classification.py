from typing import Callable
from unittest import mock

import datasets
import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
import vaex

import dataquality
import dataquality as dq
from dataquality.exceptions import GalileoException, GalileoWarning
from dataquality.loggers.data_logger.base_data_logger import DataSet
from dataquality.loggers.data_logger.text_classification import (
    TextClassificationDataLogger,
)
from dataquality.schemas.split import Split


def test_duplicate_rows(set_test_config, cleanup_after_use) -> None:
    set_test_config(task_type="text_classification")
    text_inputs = [
        "what movies star bruce willis",
        "show me films with drew barrymore from the 1980s",
        "what movies starred both al pacino and robert deniro",
        "find me all of the movies that starred harold ramis and bill murray",
        "find me a movie with a quote about baseball in it",
    ]
    gold = ["A", "C", "B", "A", "C"]
    ids = [0, 1, 2, 3, 4]

    dq.log_data_samples(texts=text_inputs, labels=gold, split="validation", ids=ids)
    dq.log_data_samples(texts=text_inputs, labels=gold, split="training", ids=ids)

    with pytest.raises(GalileoException):
        dq.log_data_samples(texts=text_inputs, labels=gold, split="validation", ids=ids)

    dq.log_data_samples(texts=text_inputs, labels=gold, split="test", ids=ids)

    with pytest.raises(GalileoException):
        dq.log_data_samples(texts=text_inputs, labels=gold, split="training", ids=ids)


def test_duplicate_output_rows(set_test_config, cleanup_after_use) -> None:
    set_test_config(task_type="text_classification")
    text_inputs = [
        "what movies star bruce willis",
        "show me films with drew barrymore from the 1980s",
        "what movies starred both al pacino and robert deniro",
        "find me all of the movies that starred harold ramis and bill murray",
        "find me a movie with a quote about baseball in it",
    ]
    gold = ["A", "C", "B", "A", "C"]
    ids = list(range(5))

    dq.log_data_samples(texts=text_inputs, labels=gold, split="validation", ids=ids)
    dq.log_data_samples(texts=text_inputs, labels=gold, split="training", ids=ids)

    embs = np.random.rand(5, 100)
    logits = np.random.rand(5, 100)
    ids = list(range(5))
    dq.log_model_outputs(embs=embs, logits=logits, ids=ids, split="training", epoch=0)
    dq.log_model_outputs(embs=embs, logits=logits, ids=ids, split="training", epoch=0)

    with pytest.raises(GalileoException) as e:
        dq.get_data_logger().upload()

    assert str(e.value).startswith("It seems your logged output data has duplicate ids")


def test_log_data_sample(
    set_test_config: Callable, cleanup_after_use: Callable
) -> None:
    logger = TextClassificationDataLogger()
    with mock.patch("dataquality.core.log.get_data_logger") as mock_method:
        mock_method.return_value = logger
        dq.log_data_sample(text="sample 1", label="A", id=1, split="training")

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
                "my_labels": ["A", "A", "B"],
                "my_id": [1, 2, 3],
            }
        ),
        vaex.from_dict(
            {
                "my_text": ["sample1", "sample2", "sample3"],
                "my_labels": ["A", "A", "B"],
                "my_id": [1, 2, 3],
            }
        ),
        [
            {"my_text": "sample1", "my_labels": "A", "my_id": 1},
            {"my_text": "sample2", "my_labels": "A", "my_id": 2},
            {"my_text": "sample3", "my_labels": "B", "my_id": 3},
        ],
        tf.data.Dataset.from_tensor_slices(
            {
                "my_text": ["sample1", "sample2", "sample3"],
                "my_labels": ["A", "A", "B"],
                "my_id": [1, 2, 3],
            }
        ),
        datasets.Dataset.from_dict(
            dict(
                my_text=["sample1", "sample2", "sample3"],
                my_labels=["A", "A", "B"],
                my_id=[1, 2, 3],
            )
        ),
    ],
)
def test_log_dataset(
    dataset: DataSet, set_test_config: Callable, cleanup_after_use: Callable
) -> None:
    logger = TextClassificationDataLogger()

    with mock.patch("dataquality.core.log.get_data_logger") as mock_method:
        mock_method.return_value = logger
        dq.log_dataset(
            dataset, text="my_text", label="my_labels", id="my_id", split="train"
        )

        assert logger.texts == ["sample1", "sample2", "sample3"]
        assert logger.labels == ["A", "A", "B"]
        assert logger.ids == [1, 2, 3]
        assert logger.split == Split.training


@pytest.mark.parametrize(
    "dataset",
    [
        [
            ("sample1", "A", "ID1"),
            ("sample2", "A", "ID2"),
            ("sample3", "B", "ID3"),
        ],
        tf.data.Dataset.from_tensor_slices(
            [
                ("sample1", "A", "ID1"),
                ("sample2", "A", "ID2"),
                ("sample3", "B", "ID3"),
            ]
        ),
    ],
)
def test_log_dataset_tuple(
    dataset: DataSet, set_test_config: Callable, cleanup_after_use: Callable
) -> None:
    logger = TextClassificationDataLogger()

    with mock.patch("dataquality.core.log.get_data_logger") as mock_method:
        mock_method.return_value = logger
        dq.log_dataset(dataset, text=0, label=1, id=2, split="train")

        assert logger.texts == ["sample1", "sample2", "sample3"]
        assert logger.labels == ["A", "A", "B"]
        assert logger.ids == ["ID1", "ID2", "ID3"]
        assert logger.split == Split.training


@pytest.mark.parametrize(
    "dataset",
    [
        [
            {"text": "sample1", "label": "A", "id": 1},
            {"text": "sample2", "label": "A", "id": 2},
            {"text": "sample3", "label": "B", "id": 3},
        ],
    ],
)
def test_log_dataset_default(
    dataset: DataSet, set_test_config: Callable, cleanup_after_use: Callable
) -> None:
    """Tests that the default keys work as expected when not passed in"""
    logger = TextClassificationDataLogger()
    with mock.patch("dataquality.core.log.get_data_logger") as mock_method:
        mock_method.return_value = logger
        dq.log_dataset(dataset, split="train")

        assert logger.texts == ["sample1", "sample2", "sample3"]
        assert logger.labels == ["A", "A", "B"]
        assert logger.ids == [1, 2, 3]
        assert logger.split == Split.training


@pytest.mark.parametrize(
    "dataset",
    [
        [
            {"text": "sample1", "label": "A", "id": 1},
            {"text": "sample2", "label": "A", "id": 2},
            {"text": "sample3", "label": "B", "id": 3},
        ],
    ],
)
def test_output_no_input_for_split(
    dataset: DataSet, set_test_config: Callable, cleanup_after_use: Callable
) -> None:
    """Validates that no error is thrown and a warning is printed on finish

    When you try to upload data for a split that has no input data
    """
    dq.log_dataset(dataset, split="train")
    dq.log_model_outputs(
        embs=np.random.rand(3, 50),
        logits=np.random.rand(3, 10),
        ids=[1, 2, 3],
        split="validation",
        epoch=0,
    )
    with pytest.warns(GalileoWarning, match=r"There was output data"):
        dq.get_data_logger().upload()


@pytest.mark.parametrize("set_labels_first", [True, False])
def test_logged_labels_dont_match_set_labels(
    set_labels_first: bool, set_test_config: Callable, cleanup_after_use: Callable
) -> None:
    """An error should be thrown when the set labels dont match the logged labels"""
    set_test_config(task_type="text_classification")
    labels = ["A", "B", "C"]
    # labels are the index, not the actual labels. No good
    dataset = [
        {"text": "sample1", "label": 1, "id": 1},
        {"text": "sample2", "label": 1, "id": 2},
        {"text": "sample3", "label": 2, "id": 3},
    ]
    if set_labels_first:
        dataquality.set_labels_for_run(labels)
        with pytest.raises(AssertionError) as e:
            dataquality.log_dataset(dataset, split="train")
    else:
        dataquality.log_dataset(dataset, split="train")
        dataquality.get_data_logger().logger_config.observed_num_labels = len(labels)
        dataquality.set_labels_for_run(labels)
        with pytest.raises(AssertionError) as e:
            dataquality.get_data_logger().validate_labels()
    assert str(e.value).startswith("Labels set to")


def test_log_int_labels(set_test_config: Callable, cleanup_after_use: Callable) -> None:
    """Tests that a user can log their labels as ints without issue"""
    set_test_config(task_type="text_classification")
    labels = [1, 2, 3]
    # labels are the index, not the actual labels. No good
    dataset = pd.DataFrame(
        [
            {"text": "sample1", "label": 1, "id": 1},
            {"text": "sample2", "label": 1, "id": 2},
            {"text": "sample3", "label": 2, "id": 3},
        ]
    )
    dataquality.log_dataset(dataset, split="train")
    dataquality.get_data_logger().logger_config.observed_num_labels = len(labels)

    dataquality.set_labels_for_run(labels)
    # Should not throw an error
    dataquality.get_data_logger().validate_labels()
    assert sorted(
        dataquality.get_data_logger().logger_config.observed_labels
    ) == sorted(["1", "2"])
    assert sorted(dataquality.get_data_logger().logger_config.labels) == sorted(
        ["1", "2", "3"]
    )
