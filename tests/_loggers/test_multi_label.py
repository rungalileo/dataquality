from random import randint, sample
from typing import Callable, Tuple
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import vaex

import dataquality
import dataquality as dq
from dataquality.exceptions import GalileoException
from dataquality.loggers.data_logger.base_data_logger import DataSet
from dataquality.loggers.data_logger.text_multi_label import TextMultiLabelDataLogger
from dataquality.loggers.model_logger.text_multi_label import TextMultiLabelModelLogger
from dataquality.schemas.split import Split


def test_duplicate_rows(set_test_config, cleanup_after_use) -> None:
    set_test_config(task_type="text_multi_label")
    dq.get_data_logger().logger_config.binary = False
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
    dq.get_data_logger().logger_config.binary = False
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

    assert str(e.value).startswith("It seems your logged output data has duplicate ids")


def test_log_data_sample(
    set_test_config: Callable, cleanup_after_use: Callable
) -> None:
    set_test_config(task_type="text_multi_label")
    dq.get_data_logger().logger_config.binary = False
    logger = TextMultiLabelDataLogger()
    with mock.patch("dataquality.core.log.get_data_logger") as mock_method:
        mock_method.return_value = logger
        dq.log_data_sample(text="sample 1", task_labels="A", id=1, split="training")

        assert logger.texts == ["sample 1"]
        assert logger.labels == [["A"]]
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
    logger.logger_config.binary = False

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
    logger.logger_config.binary = False
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


def test_log_data_binary(
    set_test_config: Callable, cleanup_after_use: Callable
) -> None:
    set_test_config(task_type="text_multi_label")
    logger = TextMultiLabelDataLogger()
    with mock.patch("dataquality.core.log.get_data_logger") as mock_method:
        mock_method.return_value = logger
        tasks = ["A", "B", "C", "D"]
        dq.set_tasks_for_run(tasks, binary=True)
        task_labels = [sample(tasks, k=randint(1, 4)) for _ in range(10)]
        dq.log_data_samples(
            texts=[f"sample {i}" for i in range(10)],
            task_labels=task_labels,
            ids=list(range(10)),
            split="training",
        )

        assert logger.texts == [f"sample {i}" for i in range(10)]
        assert logger.ids == list(range(10))
        for logged_labels, clean_sample_labels in zip(task_labels, logger.labels):
            assert len(clean_sample_labels) == 4
            for task in tasks:
                if task in logged_labels:
                    assert task in clean_sample_labels
                else:
                    assert f"NOT_{task}" in clean_sample_labels
        assert logger.logger_config.observed_num_tasks == 4


def test_log_data_binary_not_setting_binary(
    set_test_config: Callable, cleanup_after_use: Callable
) -> None:
    set_test_config(task_type="text_multi_label")
    dq.get_data_logger().logger_config.binary = False

    tasks = ["A", "B", "C", "D"]
    task_labels = [sample(tasks, k=randint(1, 4)) for _ in range(10)]
    with pytest.raises(AssertionError) as e:
        dq.log_data_samples(
            texts=[f"sample {i}" for i in range(10)],
            task_labels=task_labels,
            ids=list(range(10)),
            split="training",
        )
    # We didn't set binary=true in our tasks so during logging we should get a nice err
    assert str(e.value).startswith(
        "Each training input must have the same number of labels."
    )
    assert "If this is a binary multi label and" in str(e.value)


def test_set_tasks_not_set_binary(
    set_test_config: Callable, cleanup_after_use: Callable
) -> None:
    set_test_config(task_type="text_multi_label")

    tasks = ["A", "B", "C", "D"]
    dq.set_tasks_for_run(tasks, binary=False)
    # We didn't set binary=True above so validation should fail nicely
    with pytest.raises(ValueError) as e:
        dq.set_labels_for_run(tasks)

    err = e.value.errors()[0]["msg"]
    assert err.startswith("Labels must be a list of lists.")
    assert "If you are running a binary multi-label case," in err


def test_log_model_outputs_binary(
    set_test_config: Callable, cleanup_after_use: Callable
) -> None:
    set_test_config(task_type="text_multi_label")
    tasks = ["A", "B", "C", "D"]
    dq.set_tasks_for_run(tasks, binary=True)

    logger = TextMultiLabelModelLogger(
        embs=np.random.rand(10, 100),  # 10 samples, 100 emb per sample
        logits=np.random.rand(10, 5),  # 10 samples, 5 tasks
        ids=list(range(10)),
        split="training",
        epoch=0,
    )
    logger.logger_config.observed_num_tasks = 5
    logger._log()

    assert not hasattr(logger, "logits")
    assert logger.probs.shape == (10, 5, 2)  # samples X tasks X classes per task
    for sample_probs in logger.probs:
        for task_probs in sample_probs:
            assert np.isclose(np.sum(task_probs), 1.0)


@pytest.mark.parametrize("dims", [(10, 5, 2), (10,)])
def test_log_model_outputs_binary_bad_shapes(
    dims: Tuple, set_test_config: Callable, cleanup_after_use: Callable
) -> None:
    set_test_config(task_type="text_multi_label")
    tasks = ["A", "B", "C", "D"]
    dq.set_tasks_for_run(tasks, binary=True)

    logger = TextMultiLabelModelLogger(
        embs=np.random.rand(10, 100),  # 10 samples, 100 emb per sample
        logits=np.random.rand(*dims),
        ids=list(range(10)),
        split="training",
        epoch=0,
    )
    ndim = 5 if len(dims) == 3 else 2
    logger.logger_config.observed_num_tasks = ndim

    with pytest.raises(GalileoException) as e:
        logger._log()

    err = str(e.value)
    if len(dims) == 1:
        assert "Probs/logits must have at least 2 dimensions, they have 1" in err
    else:
        assert (
            f"In binary multi-label, your logits should have 2 dimensions, but they "
            f"currently have {len(dims)}."
        ) in err


@pytest.mark.parametrize("set_labels_first", [True, False])
def test_logged_labels_dont_match_set_labels(
    set_labels_first: bool, set_test_config: Callable, cleanup_after_use: Callable
) -> None:
    """An error should be thrown when the set labels dont match the logged labels"""
    set_test_config(task_type="text_multi_label")
    dq.get_data_logger().logger_config.binary = False
    labels = [["A", "NOT_A"], ["B", "NOT_B"], ["C", "NOT_C"]]
    # labels are the index, not the actual labels. No good
    dataset = pd.DataFrame(
        {
            "text": ["sample1", "sample2", "sample3"],
            "label": [[1, 1, 2], [3, 1, 2], [2, 3, 2]],
            "id": [1, 2, 3],
        }
    )
    if set_labels_first:
        dataquality.set_labels_for_run(labels)
        with pytest.raises(AssertionError) as e:
            dataquality.log_dataset(dataset, split="train")
        assert str(e.value).startswith(
            "The input labels you log must be exactly the same"
        )
    else:
        dataquality.log_dataset(dataset, split="train")
        dataquality.get_data_logger().logger_config.observed_num_labels = [2, 2, 2]
        dataquality.set_labels_for_run(labels)
        with pytest.raises(AssertionError) as e:
            dataquality.get_data_logger().validate_labels()
        assert str(e.value).startswith("The labels set for task #")
