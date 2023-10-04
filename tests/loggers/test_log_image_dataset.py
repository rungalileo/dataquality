import os.path
from tempfile import TemporaryDirectory
from typing import Callable, Generator
from unittest.mock import MagicMock, patch

import vaex
from datasets import load_dataset
from torchvision.datasets import ImageFolder

import dataquality as dq
from dataquality.schemas.cv import GAL_LOCAL_IMAGES_PATHS
from dataquality.utils.thread_pool import ThreadPoolManager
from tests.assets.constants import TEST_IMAGES_FOLDER_ROOT
from tests.conftest import TestSessionVariables
from tests.test_utils.mock_data import MockDatasetCV

mnist_dataset = load_dataset("mnist", split="train").select(range(20))

TESTING_DATASETS = {
    "mnist": dict(
        dataset=mnist_dataset,
        labels=mnist_dataset.features["label"].names,
        imgs_colname="image",
    )
}


@patch.object(dq.core.finish, "upload_dq_log_file")
@patch.object(dq.clients.api.ApiClient, "make_request")
@patch.object(dq.clients.objectstore.ObjectStore, "create_object")
def test_with_pd_local_only(
    mock_create_object: MagicMock,
    mock_make_request: MagicMock,
    mock_upload_log_file: MagicMock,
    set_test_config: Callable,
    cleanup_after_use: Generator,
    test_session_vars: TestSessionVariables,
) -> None:
    """
    Test logging the data with a pd when only local data is provided
    """
    set_test_config(task_type="image_classification")

    cvdata = MockDatasetCV()
    imgs_local_colname = "image_path"

    dq.set_labels_for_run(cvdata.labels)
    dq.log_image_dataset(
        dataset=cvdata.dataframe,
        imgs_local_colname=imgs_local_colname,
        split="training",
    )

    # read logged data
    ThreadPoolManager.wait_for_threads()
    df = vaex.open(f"{test_session_vars.LOCATION}/input_data/training/*.arrow")

    # assert that the df contains the remote images under "text" (minio rel path)
    minio_path = df["text"].tolist()[0]
    assert minio_path.count("/") == 1 and minio_path.split(".")[-1] == "png"
    # assert that the saved df also contains the local images in the specified column
    assert (
        df[GAL_LOCAL_IMAGES_PATHS].tolist()[0]
        == cvdata.dataframe.loc[0, imgs_local_colname]
    )
    assert os.path.isfile(df[GAL_LOCAL_IMAGES_PATHS].tolist()[0])


def test_with_pd_remote_only(
    set_test_config: Callable,
    cleanup_after_use: Generator,
    test_session_vars: TestSessionVariables,
) -> None:
    """
    Test logging the data with a pd when only remote data is provided
    """
    set_test_config(task_type="image_classification")

    cvdata = MockDatasetCV()
    imgs_remote_colname = "remote_image_path"

    dq.set_labels_for_run(cvdata.labels)
    dq.log_image_dataset(
        dataset=cvdata.dataframe,
        imgs_remote=imgs_remote_colname,
        split="training",
    )

    # read logged data
    ThreadPoolManager.wait_for_threads()
    df = vaex.open(f"{test_session_vars.LOCATION}/input_data/training/*.arrow")

    # assert that the saved df contains the remote images in the specified column
    assert df["text"].tolist()[0] == cvdata.dataframe.loc[0, imgs_remote_colname]


def test_with_pd_local_and_remote(
    set_test_config: Callable,
    cleanup_after_use: Generator,
    test_session_vars: TestSessionVariables,
) -> None:
    """
    Test logging the data with a pd when local and remote data are provided
    """
    set_test_config(task_type="image_classification")

    cvdata = MockDatasetCV()
    imgs_local_colname = "image_path"
    imgs_remote_colname = "remote_image_path"

    dq.set_labels_for_run(cvdata.labels)
    dq.log_image_dataset(
        dataset=cvdata.dataframe,
        imgs_local_colname=imgs_local_colname,
        imgs_remote=imgs_remote_colname,
        split="training",
    )

    # read logged data
    ThreadPoolManager.wait_for_threads()
    df = vaex.open(f"{test_session_vars.LOCATION}/input_data/training/*.arrow")

    # assert that the saved df contains the local images in the specified column
    assert (
        df[GAL_LOCAL_IMAGES_PATHS].tolist()[0]
        == cvdata.dataframe.loc[0, imgs_local_colname]
    )
    assert os.path.isfile(df[GAL_LOCAL_IMAGES_PATHS].tolist()[0])
    # assert that the saved df contains the remote images under "text"
    assert df["text"].tolist()[0] == cvdata.dataframe.loc[0, imgs_remote_colname]


@patch.object(dq.clients.objectstore.ObjectStore, "create_object")
def test_with_hf_local_only_images(
    mock_create_object: MagicMock,
    set_test_config: Callable,
    cleanup_after_use: Generator,
    test_session_vars: TestSessionVariables,
) -> None:
    """
    Test logging the data with HF when only local data is provided (as images).
    We don't have local paths in this case.
    """
    set_test_config(task_type="image_classification")
    imgs_local_colname = "image"

    with TemporaryDirectory():
        dataset_info = TESTING_DATASETS["mnist"]
        dataset = dataset_info["dataset"]

        dq.set_labels_for_run(dataset_info["labels"])
        dq.log_image_dataset(
            dataset=dataset,
            label="label",
            imgs_local_colname=imgs_local_colname,
            split="training",
        )

        # read logged data
        ThreadPoolManager.wait_for_threads()
        df = vaex.open(f"{test_session_vars.LOCATION}/input_data/training/*.arrow")

        # assert that the df contains the remote images under "text" (minio rel path)
        minio_path = df["text"].tolist()[0]
        assert minio_path.count("/") == 1 and minio_path.split(".")[-1] == "png"


@patch.object(dq.clients.objectstore.ObjectStore, "create_object")
def test_with_hf_local_only_paths(
    mock_create_object: MagicMock,
    set_test_config: Callable,
    cleanup_after_use: Generator,
    test_session_vars: TestSessionVariables,
) -> None:
    """
    Test logging the data with HF when only local data is provided (as paths)
    """
    set_test_config(task_type="image_classification")
    imgs_local_colname = "image_path"

    with TemporaryDirectory() as imgs_dir:

        def save_and_record_path(example, index):
            path = os.path.join(imgs_dir, f"{index:04d}.jpg")
            example["image"].save(path)
            return {imgs_local_colname: path, **example}

        dataset_info = TESTING_DATASETS["mnist"]
        dataset = dataset_info["dataset"]
        dataset_with_paths = dataset.map(save_and_record_path, with_indices=True)

        dq.set_labels_for_run(dataset_info["labels"])
        dq.log_image_dataset(
            dataset=dataset_with_paths,
            imgs_local_colname=imgs_local_colname,
            split="training",
        )

        # read logged data
        ThreadPoolManager.wait_for_threads()
        df = vaex.open(f"{test_session_vars.LOCATION}/input_data/training/*.arrow")

        # assert that the df contains the remote images under "text" (minio rel path)
        minio_path = df["text"].tolist()[0]
        assert minio_path.count("/") == 1 and minio_path.split(".")[-1] == "jpg"
        # assert that the saved df contains the local images in the specified column
        assert (
            df[GAL_LOCAL_IMAGES_PATHS].tolist()[0]
            == dataset_with_paths[imgs_local_colname][0]
        )
        assert os.path.isfile(df[GAL_LOCAL_IMAGES_PATHS].tolist()[0])


def test_with_hf_local_and_remote(
    set_test_config: Callable,
    cleanup_after_use: Generator,
    test_session_vars: TestSessionVariables,
) -> None:
    """
    Test logging the data with HF when local data is provided (as paths) + remote paths
    """
    set_test_config(task_type="image_classification")
    imgs_local_colname = "image_path"
    imgs_remote_colname = "remote_image_path"
    fake_remote_base = "s3://some_bucket/some_dir"

    with TemporaryDirectory() as imgs_dir:

        def save_and_record_path(example, index):
            image_name = f"{index:04d}.jpg"
            path = os.path.join(imgs_dir, image_name)
            remote_path = f"{fake_remote_base}/{image_name}"
            example["image"].save(path)
            return {
                imgs_local_colname: path,
                imgs_remote_colname: remote_path,
                **example,
            }

        dataset_info = TESTING_DATASETS["mnist"]
        dataset = dataset_info["dataset"]
        dataset_with_paths = dataset.map(save_and_record_path, with_indices=True)

        dq.set_labels_for_run(dataset_info["labels"])
        dq.log_image_dataset(
            dataset=dataset_with_paths,
            imgs_local_colname=imgs_local_colname,
            imgs_remote=imgs_remote_colname,
            split="training",
        )

        # read logged data
        ThreadPoolManager.wait_for_threads()
        df = vaex.open(f"{test_session_vars.LOCATION}/input_data/training/*.arrow")

        # assert that the saved df contains the remote images under "text"
        assert df["text"].tolist()[0] == dataset_with_paths[0].get(imgs_remote_colname)
        # assert that the saved df contains the local images in the specified column
        assert df[GAL_LOCAL_IMAGES_PATHS].tolist()[0] == dataset_with_paths[0].get(
            imgs_local_colname
        )
        assert os.path.isfile(df[GAL_LOCAL_IMAGES_PATHS].tolist()[0])


@patch.object(dq.core.finish, "upload_dq_log_file")
@patch.object(dq.clients.api.ApiClient, "make_request")
@patch.object(dq.clients.objectstore.ObjectStore, "create_object")
def test_with_ImageFolder_local_only(
    mock_create_object: MagicMock,
    mock_make_request: MagicMock,
    mock_upload_log_file: MagicMock,
    set_test_config: Callable,
    cleanup_after_use: Generator,
    test_session_vars: TestSessionVariables,
) -> None:
    """
    Test logging the data with Imagefolder when only local data is provided
    """
    set_test_config(task_type="image_classification")

    train_dataset = ImageFolder(root=TEST_IMAGES_FOLDER_ROOT)

    dq.set_labels_for_run(train_dataset.classes)
    dq.log_image_dataset(
        dataset=train_dataset,
        imgs_local="randommmm",
        split="training",
    )

    # read logged data
    ThreadPoolManager.wait_for_threads()
    df = vaex.open(f"{test_session_vars.LOCATION}/input_data/training/*.arrow")

    # assert that the df contains the remote images under "text" (minio rel path)
    minio_path = df["text"].tolist()[0]
    assert minio_path.count("/") == 1 and minio_path.split(".")[-1] == "png"
    # assert that the saved df also contains the local images in the specified column
    assert df[GAL_LOCAL_IMAGES_PATHS].tolist()[0] == train_dataset.samples[0][0]
    assert os.path.isfile(df[GAL_LOCAL_IMAGES_PATHS].tolist()[0])


@patch.object(dq.core.finish, "upload_dq_log_file")
@patch.object(dq.clients.api.ApiClient, "make_request")
@patch.object(dq.clients.objectstore.ObjectStore, "create_object")
def test_with_ImageFolder_local_and_remote(
    mock_create_object: MagicMock,
    mock_make_request: MagicMock,
    mock_upload_log_file: MagicMock,
    set_test_config: Callable,
    cleanup_after_use: Generator,
    test_session_vars: TestSessionVariables,
) -> None:
    """
    Test logging the data with Imagefolder when local and remote data are provided
    """
    set_test_config(task_type="image_classification")

    train_dataset = ImageFolder(root=TEST_IMAGES_FOLDER_ROOT)
    imgs_remote = "s3://some_bucket/some_dir"

    dq.set_labels_for_run(train_dataset.classes)
    dq.log_image_dataset(
        dataset=train_dataset,
        imgs_remote=imgs_remote,
        split="training",
    )

    # read logged data
    ThreadPoolManager.wait_for_threads()
    df = vaex.open(f"{test_session_vars.LOCATION}/input_data/training/*.arrow")

    # assert that the saved df contains the local images in the specified column
    assert df[GAL_LOCAL_IMAGES_PATHS].tolist()[0] == train_dataset.samples[0][0]
    assert os.path.isfile(df[GAL_LOCAL_IMAGES_PATHS].tolist()[0])
    # assert that the saved df contains the remote images under "text"
    assert df["text"].tolist()[0].startswith(imgs_remote)
    assert df["text"].tolist()[0].endswith(train_dataset.samples[0][0].split("/")[-1])
