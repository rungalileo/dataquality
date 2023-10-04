import os.path
from tempfile import TemporaryDirectory
from typing import Callable, Generator
from unittest import mock
from unittest.mock import MagicMock, patch

import numpy as np
import vaex
from datasets import load_dataset
from torchvision.datasets import ImageFolder

import dataquality
import dataquality as dq
from dataquality.loggers.data_logger.image_classification import (
    ImageClassificationDataLogger,
)
from dataquality.schemas.cv import GAL_LOCAL_IMAGES_PATHS
from dataquality.utils.thread_pool import ThreadPoolManager
from dataquality.utils.vaex import validate_unique_ids
from tests.assets.constants import TEST_IMAGES_FOLDER_ROOT
from tests.conftest import TestSessionVariables
from tests.test_utils.mock_data import MockDatasetCV

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
    set_test_config, cleanup_after_use, test_session_vars: TestSessionVariables
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
        df = vaex.open(f"{test_session_vars.LOCATION}/{split}/0/*.hdf5")
        assert len(df) == 5
        validate_unique_ids(df, "epoch")
        df = vaex.open(f"{test_session_vars.LOCATION}/{split}/1/*.hdf5")
        assert len(df) == 5
        validate_unique_ids(df, "epoch")


def test_duplicate_ids_augmented(
    set_test_config, cleanup_after_use, test_session_vars: TestSessionVariables
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
        dq.log_model_outputs(
            embs=embs,
            logits=logits,
            ids=ids,
        )

    ThreadPoolManager.wait_for_threads()
    for split in ["training", "validation", "test"]:
        df = vaex.open(f"{test_session_vars.LOCATION}/{split}/0/*.hdf5")
        assert len(df) == 5
        validate_unique_ids(df, "epoch")
        df = vaex.open(f"{test_session_vars.LOCATION}/{split}/1/*.hdf5")
        assert len(df) == 5
        validate_unique_ids(df, "epoch")


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
    set_test_config: callable,
    cleanup_after_use: callable,
    test_session_vars: TestSessionVariables,
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
        df = vaex.open(f"{test_session_vars.LOCATION}/{split}/0/*.hdf5")
        assert len(df) == 5
        validate_unique_ids(df, "epoch")
        df = vaex.open(f"{test_session_vars.LOCATION}/{split}/1/*.hdf5")
        assert len(df) == 5
        validate_unique_ids(df, "epoch")

    # assert that the logger config's observed ids are set
    # 6 because 3 splits * 2 epochs
    assert len(dq.get_model_logger().logger_config.observed_ids) == 6

    dq.finish()

    # assert that the logger config's observed ids are reset
    assert len(dq.get_model_logger().logger_config.observed_ids) == 0


def _test_hf_image_dataset(name, test_session_vars: TestSessionVariables) -> None:
    """
    Tests that dq.log_image_dataset can handle HF dataset inputs.
    """

    dataset_info = TESTING_DATASETS[name]

    dq.set_labels_for_run(dataset_info["labels"])
    dq.log_image_dataset(
        dataset=dataset_info["dataset"],
        label="label",
        imgs_local_colname=dataset_info["imgs_colname"],
        split="training",
    )

    # read logged data
    ThreadPoolManager.wait_for_threads()
    df = vaex.open(f"{test_session_vars.LOCATION}/input_data/training/*.arrow")

    assert len(df) == len(food_dataset)


@mock.patch("dataquality.clients.objectstore.ObjectStore.create_object")
def test_hf_dataset_food(
    mock_create_object: mock.MagicMock,
    cleanup_after_use,
    set_test_config,
    test_session_vars: TestSessionVariables,
) -> None:
    set_test_config(task_type="image_classification")
    _test_hf_image_dataset(name="food", test_session_vars=test_session_vars)


@mock.patch("dataquality.clients.objectstore.ObjectStore.create_object")
def test_hf_dataset_mnist(
    mock_create_object: mock.MagicMock,
    cleanup_after_use,
    set_test_config,
    test_session_vars: TestSessionVariables,
) -> None:
    set_test_config(task_type="image_classification")
    _test_hf_image_dataset(name="mnist", test_session_vars=test_session_vars)


@mock.patch("dataquality.clients.objectstore.ObjectStore.create_object")
def test_hf_dataset_cifar10(
    mock_create_object: mock.MagicMock,
    cleanup_after_use,
    set_test_config,
    test_session_vars: TestSessionVariables,
) -> None:
    set_test_config(task_type="image_classification")
    _test_hf_image_dataset(name="cifar10", test_session_vars=test_session_vars)


@mock.patch("dataquality.clients.objectstore.ObjectStore.create_object")
def test_hf_image_dataset_with_paths(
    mock_create_object: mock.MagicMock,
    set_test_config,
    cleanup_after_use,
    test_session_vars: TestSessionVariables,
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
            imgs_local_colname="path",
            split="training",
        )

        # read logged data
        ThreadPoolManager.wait_for_threads()
        df = vaex.open(f"{test_session_vars.LOCATION}/input_data/training/*.arrow")

        assert len(df) == len(food_dataset)


def test_prepare_df_from_ImageFolder() -> None:
    """
    Check the format of the dataframe returned by
    ImageClassificationDataLogger._prepare_df_from_ImageFolder, called in
    dq.log_image_dataset
    """

    image_logger = ImageClassificationDataLogger()

    train_dataset = ImageFolder(root=TEST_IMAGES_FOLDER_ROOT)

    df = image_logger._prepare_df_from_ImageFolder(dataset=train_dataset)

    # Assert that the dataframe is how we'd expect it to be by looking at the folder
    assert set(df.columns) == {"id", "label", GAL_LOCAL_IMAGES_PATHS}
    assert len(df) == 5
    assert set(df.label.unique()) == {"labelA", "labelB"}
    assert set(df.id.unique()) == set(range(5))
    assert df.loc[0, "gal_local_images_paths"].endswith(".png")


def test_prepare_df_from_ImageFolder_with_remote_imgs() -> None:
    """
    Check the format of the dataframe returned by
    ImageClassificationDataLogger._prepare_df_from_ImageFolder, called in
    dq.log_image_dataset, when using remote images
    """

    image_logger = ImageClassificationDataLogger()

    train_dataset = ImageFolder(root=TEST_IMAGES_FOLDER_ROOT)
    imgs_remote_location = "gs://some_biiiig_bucket"

    df = image_logger._prepare_df_from_ImageFolder(
        dataset=train_dataset, imgs_remote_location=imgs_remote_location
    )

    # Assert that the dataframe is how we'd expect it to be by looking at the folder
    assert set(df.columns) == {
        "id",
        "text",
        "label",
        GAL_LOCAL_IMAGES_PATHS,
    }
    assert len(df) == 5
    assert set(df.label.unique()) == {"labelA", "labelB"}
    assert set(df.id.unique()) == set(range(5))
    assert df.loc[0, "text"].startswith(imgs_remote_location)
    assert df.loc[0, "text"].endswith(".png")


def test_prepare_df_from_ImageFolder_inference() -> None:
    """
    Check the format of the dataframe returned by
    ImageClassificationDataLogger._prepare_df_from_ImageFolder, called in
    dq.log_image_dataset
    """

    image_logger = ImageClassificationDataLogger()

    train_dataset = ImageFolder(root=TEST_IMAGES_FOLDER_ROOT)

    df = image_logger._prepare_df_from_ImageFolder(
        dataset=train_dataset, split="inference"
    )

    # Assert that the dataframe is how we'd expect it to be by looking at the folder
    # with no labels
    assert set(df.columns) == {"id", GAL_LOCAL_IMAGES_PATHS}
    assert len(df) == 5
    assert set(df.id.unique()) == set(range(5))
    assert df.loc[0, GAL_LOCAL_IMAGES_PATHS].endswith(".png")


@patch.object(dq.core.finish, "upload_dq_log_file")
@patch.object(dq.clients.api.ApiClient, "make_request")
@patch.object(dq.core.finish, "wait_for_run")
def test_smart_features(
    mock_wait_for_run: MagicMock,
    mock_make_request: MagicMock,
    mock_upload_log_file: MagicMock,
    set_test_config: Callable,
    cleanup_after_use: Generator,
    test_session_vars: TestSessionVariables,
) -> None:
    set_test_config(task_type="image_classification")

    # Create Mock data and Log the input images
    df_train = MockDatasetCV()
    dq.set_labels_for_run(df_train.labels)
    dq.log_image_dataset(
        dataset=df_train.dataframe,
        label="label",
        imgs_local_colname="image_path",
        split="training",
    )

    ThreadPoolManager.wait_for_threads()

    # Test the CV Smart features
    image_classification_logger = dq.get_data_logger()

    in_frame_path = f"{image_classification_logger.input_data_path}/training"
    in_frame_split = vaex.open(
        f"{in_frame_path}/*.{image_classification_logger.INPUT_DATA_FILE_EXT}"
    )

    in_frame_split = image_classification_logger.add_cv_smart_features(
        in_frame_split, "training"
    )

    outlier_cols = {
        "is_near_duplicate",
        "near_duplicate_id",
        "is_blurry",
        "is_underexposed",
        "is_overexposed",
        "has_low_contrast",
        "has_low_content",
        "has_odd_size",
        "has_odd_ratio",
        "has_odd_channels",
    }
    assert outlier_cols.issubset(in_frame_split.get_column_names())
