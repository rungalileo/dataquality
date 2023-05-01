import os.path
from tempfile import TemporaryDirectory
from unittest import mock
from unittest.mock import MagicMock

import numpy as np
import vaex
from datasets import load_dataset

import dataquality
import dataquality as dq
from dataquality.utils.thread_pool import ThreadPoolManager
from dataquality.utils.vaex import validate_unique_ids
from tests.conftest import LOCATION

food_dataset = load_dataset("sasha/dog-food", split="train")
food_dataset = food_dataset.select(range(20))

mnist_dataset = load_dataset("mnist", split="train")
mnist_dataset = mnist_dataset.select(range(20))

cifar10_dataset = load_dataset("cifar10", split="train")
cifar10_dataset = cifar10_dataset.select(range(20))

TESTING_DATASETS = {
    "food": dict(
        dataset=food_dataset,
        labels=food_dataset.features["label"].names,
        imgs_colname="image",
    ),
    "mnist": dict(
        dataset=mnist_dataset,
        labels=mnist_dataset.features["label"].names,
        imgs_colname="image",
    ),
    "cifar10": dict(
        dataset=cifar10_dataset,
        labels=cifar10_dataset.features["label"].names,
        imgs_colname="img",
    ),
}


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


def _test_hf_image_dataset(name) -> None:
    """
    Tests that dq.log_image_dataset can handle HF dataset inputs.
    """

    dataset_info = TESTING_DATASETS[name]

    dq.set_labels_for_run(dataset_info["labels"])
    dq.log_image_dataset(
        dataset=dataset_info["dataset"],
        label="label",
        imgs_colname=dataset_info["imgs_colname"],
        split="training",
    )

    # read logged data
    ThreadPoolManager.wait_for_threads()
    df = vaex.open(f"{LOCATION}/input_data/training/*.arrow")

    assert len(df) == len(food_dataset)


@mock.patch("dataquality.clients.objectstore.ObjectStore.create_object")
def test_hf_dataset_food(
    mock_create_object: mock.MagicMock,
    cleanup_after_use,
    set_test_config,
) -> None:
    set_test_config(task_type="image_classification")
    _test_hf_image_dataset("food")


@mock.patch("dataquality.clients.objectstore.ObjectStore.create_object")
def test_hf_dataset_mnist(
    mock_create_object: mock.MagicMock,
    cleanup_after_use,
    set_test_config,
) -> None:
    set_test_config(task_type="image_classification")
    _test_hf_image_dataset("mnist")


@mock.patch("dataquality.clients.objectstore.ObjectStore.create_object")
def test_hf_dataset_cifar10(
    mock_create_object: mock.MagicMock,
    cleanup_after_use,
    set_test_config,
) -> None:
    set_test_config(task_type="image_classification")
    _test_hf_image_dataset("cifar10")


@mock.patch("dataquality.clients.objectstore.ObjectStore.create_object")
def test_hf_image_dataset_with_paths(
    mock_create_object: mock.MagicMock,
    set_test_config,
    cleanup_after_use,
) -> None:
    """
    Tests that dq.log_image_dataset can handle imgs_location_colname when
    passed an HF dataset.
    """
    set_test_config(task_type="image_classification")

    with TemporaryDirectory() as imgs_dir:

        def save_and_record_path(example, index):
            path = os.path.join(imgs_dir, f"{index:04d}.jpg")
            example["image"].save(path)
            return {"path": path, **example}

        dataset_info = TESTING_DATASETS["food"]

        dataset = dataset_info["dataset"]

        dataset_with_paths = dataset.map(save_and_record_path, with_indices=True)

        dq.set_labels_for_run(dataset_info["labels"])
        dq.log_image_dataset(
            dataset=dataset_with_paths,
            label="label",
            imgs_location_colname="path",
            split="training",
        )

        # read logged data
        ThreadPoolManager.wait_for_threads()
        df = vaex.open(f"{LOCATION}/input_data/training/*.arrow")

        assert len(df) == len(food_dataset)
