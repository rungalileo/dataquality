from unittest import mock
from unittest.mock import MagicMock

import numpy as np
import vaex

import dataquality
import dataquality as dq
from dataquality.utils.thread_pool import ThreadPoolManager
from dataquality.utils.vaex import validate_unique_ids
from tests.conftest import LOCATION


def test_duplicate_ids_augmented_loop_thread(
    set_test_config, cleanup_after_use
) -> None:
    """
    This test is to ensure that duplicate ids caused by augmentation are not logged
    """
    set_test_config(task_type="image_classification")
    text_inputs = [
        "what movies star bruce willis",
        "show me films with drew barrymore from the 1980s",
        "what movies starred both al pacino and robert deniro",
        "find me all of the movies that starred harold ramis and bill murray",
        "find me a movie with a quote about baseball in it",
    ]
    gold = ["A", "C", "B", "A", "C"]
    ids = list(range(len(text_inputs)))
    embs = np.random.rand(5, 10)
    logits = [[0, 0, 1]] * 5

    dq.set_labels_for_run(["A", "B", "C"])
    for split in ["training", "validation", "test"]:
        dq.log_data_samples(texts=text_inputs, labels=gold, split=split, ids=ids)
        dq.set_split(split)
        dq.set_epoch(0)
        dq.log_model_outputs(
            embs=embs,
            logits=logits,
            ids=ids,
        )
        dq.log_model_outputs(
            embs=embs,
            logits=logits,
            ids=ids,
        )
        dq.set_epoch(1)
        dq.log_model_outputs(
            embs=embs,
            logits=logits,
            ids=ids,
        )

        ThreadPoolManager.wait_for_threads()
        df = vaex.open(f"{LOCATION}/{split}/0/*.hdf5")
        assert len(df) == 5
        validate_unique_ids(df, "epoch")
        df = vaex.open(f"{LOCATION}/{split}/1/*.hdf5")
        assert len(df) == 5
        validate_unique_ids(df, "epoch")


def test_duplicate_ids_augmented(set_test_config, cleanup_after_use) -> None:
    """
    This test is to ensure that duplicate ids caused by augmentation are not logged
    """
    set_test_config(task_type="image_classification")
    text_inputs = [
        "what movies star bruce willis",
        "show me films with drew barrymore from the 1980s",
        "what movies starred both al pacino and robert deniro",
        "find me all of the movies that starred harold ramis and bill murray",
        "find me a movie with a quote about baseball in it",
    ]
    gold = ["A", "C", "B", "A", "C"]
    ids = list(range(len(text_inputs)))
    embs = np.random.rand(5, 10)
    logits = [[0, 0, 1]] * 5

    dq.set_labels_for_run(["A", "B", "C"])
    for split in ["training", "validation", "test"]:
        dq.log_data_samples(texts=text_inputs, labels=gold, split=split, ids=ids)
        dq.set_split(split)
        dq.set_epoch(0)
        dq.log_model_outputs(
            embs=embs,
            logits=logits,
            ids=ids,
        )
        dq.log_model_outputs(
            embs=embs,
            logits=logits,
            ids=ids,
        )
        dq.set_epoch(1)
        dq.log_model_outputs(
            embs=embs,
            logits=logits,
            ids=ids,
        )
        dq.log_model_outputs(
            embs=embs,
            logits=logits,
            ids=ids,
        )

    ThreadPoolManager.wait_for_threads()
    for split in ["training", "validation", "test"]:
        df = vaex.open(f"{LOCATION}/{split}/0/*.hdf5")
        assert len(df) == 5
        validate_unique_ids(df, "epoch")
        df = vaex.open(f"{LOCATION}/{split}/1/*.hdf5")
        assert len(df) == 5
        validate_unique_ids(df, "epoch")


@mock.patch.object(
    dataquality.core.finish,
    "_version_check",
)
@mock.patch.object(
    dataquality.core.finish,
    "_reset_run",
)
@mock.patch.object(
    dataquality.core.finish,
    "upload_dq_log_file",
)
@mock.patch.object(
    dataquality.clients.api.ApiClient,
    "make_request",
)
@mock.patch.object(
    dataquality.core.finish.dataquality,
    "get_data_logger",
)
@mock.patch.object(
    dataquality.core.finish,
    "wait_for_run",
)
def test_observed_ids_cleaned_up_after_finish(
    mock_wait_for_run: MagicMock,
    mock_get_data_logger: MagicMock,
    mock_make_request: MagicMock,
    mock_upload_log_file: MagicMock,
    mock_reset_run: MagicMock,
    mock_version_check: MagicMock,
    set_test_config: callable,
    cleanup_after_use: callable,
) -> None:
    """
    This test is to ensure that duplicate ids caused by augmentation are not logged
    """
    set_test_config(task_type="image_classification")
    mock_get_data_logger.return_value = MagicMock(
        logger_config=MagicMock(conditions=None)
    )
    text_inputs = [
        "what movies star bruce willis",
        "show me films with drew barrymore from the 1980s",
        "what movies starred both al pacino and robert deniro",
        "find me all of the movies that starred harold ramis and bill murray",
        "find me a movie with a quote about baseball in it",
    ]
    gold = ["A", "C", "B", "A", "C"]
    ids = list(range(len(text_inputs)))
    embs = np.random.rand(5, 10)
    logits = [[0, 0, 1]] * 5

    dq.set_labels_for_run(["A", "B", "C"])
    for split in ["training", "validation", "test"]:
        dq.log_data_samples(texts=text_inputs, labels=gold, split=split, ids=ids)
        dq.set_split(split)
        dq.set_epoch(0)
        dq.log_model_outputs(
            embs=embs,
            logits=logits,
            ids=ids,
        )
        dq.log_model_outputs(
            embs=embs,
            logits=logits,
            ids=ids,
        )
        dq.set_epoch(1)
        dq.log_model_outputs(
            embs=embs,
            logits=logits,
            ids=ids,
        )
        dq.log_model_outputs(
            embs=embs,
            logits=logits,
            ids=ids,
        )

    ThreadPoolManager.wait_for_threads()
    for split in ["training", "validation", "test"]:
        df = vaex.open(f"{LOCATION}/{split}/0/*.hdf5")
        assert len(df) == 5
        validate_unique_ids(df, "epoch")
        df = vaex.open(f"{LOCATION}/{split}/1/*.hdf5")
        assert len(df) == 5
        validate_unique_ids(df, "epoch")

    # assert that the logger config's observed ids are set
    # 6 because 3 splits * 2 epochs
    assert len(dq.get_model_logger().logger_config.observed_ids) == 6

    dq.finish()

    # assert that the logger config's observed ids are reset
    assert len(dq.get_model_logger().logger_config.observed_ids) == 0
