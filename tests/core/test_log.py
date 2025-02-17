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
