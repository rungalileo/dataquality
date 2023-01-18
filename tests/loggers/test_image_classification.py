import os.path
from unittest import mock
from unittest.mock import MagicMock
from tempfile import TemporaryDirectory
from io import BytesIO
import base64

import numpy as np
import pandas as pd
import vaex
from PIL import Image

import dataquality
import dataquality as dq
from dataquality.utils.thread_pool import ThreadPoolManager
from dataquality.utils.vaex import validate_unique_ids
from tests.conftest import LOCATION


def test_base64(set_test_config, cleanup_after_use) -> None:
    """
    TODO
    """
    set_test_config(task_type="image_classification")

    def make_img(w, h):
        a = np.random.randint(256, size=(w, h, 3), dtype=np.uint8)
        return Image.fromarray(a)

    with TemporaryDirectory() as imgs_dir:
        images = []
        image_paths = []
        labels = []
        ids = []
        for i, xtn in enumerate(['.jpg', '.png', '.jpeg', '.gif']):
            image_path = os.path.join(imgs_dir, f"{i:.03d}.{xtn}")
            image = make_img(32, 32).save(image_path)

            images.append(image)
            image_paths.append(image_path)
            labels.append("A" if i % 2 == 0 else "B")
            ids.append(i)

        dataset = pd.DataFrame(
            dict(id=ids, label=labels, path=image_paths,)
        )

        dq.set_labels_for_run(["A", "B"])
        dq.log_image_dataset(
            dataset=dataset,
            label="label",
            imgs_location_colname="path"
        )
        ThreadPoolManager.wait_for_threads()
        df = vaex.open(f"{LOCATION}/input_data/training/data_0.arrow")

        base64_images = df["text"]
        assert len(base64_images) == len(images)

        for image, base64_image in zip(images, base64_images):
            _, _, content = base64_image.partition("base64,")
            assert len(content) > 0
            decoded = Image.open(BytesIO(base64.b64decode(content)))
            assert np.allclose(np.array(image), np.array(decoded))


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
