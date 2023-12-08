from typing import Callable

import pytest

import dataquality as dq
from dataquality.exceptions import GalileoException
from dataquality.loggers.logger_config.text_classification import (
    text_classification_logger_config as logger_config,
)


def test_set_labels_existing_run_different_labels(set_test_config: Callable) -> None:
    logger_config.existing_run = True
    logger_config.labels = ["A", "B", "C"]
    with pytest.raises(GalileoException) as e:
        dq.set_labels_for_run(["A", "B"])
    assert str(e.value).startswith("The labels provided to do match")


def test_set_labels_existing_run_same_labels(set_test_config: Callable) -> None:
    logger_config.existing_run = True
    logger_config.labels = ["A", "B", "C"]
    dq.set_labels_for_run(["A", "B", "C"])
    assert logger_config.labels == ["A", "B", "C"]


def test_set_labels_existing_run_same_labels_unsorted(
    set_test_config: Callable,
) -> None:
    logger_config.existing_run = True
    logger_config.labels = ["A", "B", "C"]
    with pytest.raises(GalileoException) as e:
        dq.set_labels_for_run(["B", "C", "A"])
    assert str(e.value).startswith("The labels provided to do match")
    assert logger_config.labels == ["A", "B", "C"] == dq.get_current_run_labels()


@pytest.mark.parametrize("label", ["hello there", "abc.def", "!!!"])
def test_set_labels_invalid_label_name(label: str, set_test_config: Callable) -> None:
    with pytest.raises(GalileoException) as e:
        dq.set_labels_for_run([label, "B"])
    assert str(e.value) == (
        f"Label `{label}` is not valid. Only alphanumeric "
        "characters, dashes, and underscores are supported."
    )


@pytest.mark.parametrize("label", ["hello there", "abc.def", "!!!"])
def test_set_labels_invalid_label_name_multi_label(
    label: str, set_test_config: Callable
) -> None:
    with pytest.raises(GalileoException) as e:
        dq.set_labels_for_run([[label, "A"], [label, "B"]])
    assert str(e.value) == (
        f"Label `{label}` is not valid. Only alphanumeric "
        "characters, dashes, and underscores are supported."
    )


@pytest.mark.parametrize("task", ["hello there", "abc.def", "!!!"])
def test_set_tasks_invalid_task_name(task: str, set_test_config: Callable) -> None:
    with pytest.raises(GalileoException) as e:
        dq.set_tasks_for_run([task, "B"])
    assert str(e.value) == (
        f"Task `{task}` is not valid. Only alphanumeric "
        "characters, dashes, and underscores are supported."
    )
