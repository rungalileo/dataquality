from pathlib import Path
from typing import Callable, Generator
from unittest.mock import MagicMock, patch
from glob import glob
import dataquality as dq
from dataquality.integrations.fastai import FastAiDQCallback
from dataquality.integrations.torch import unwatch, watch
from dataquality.schemas.task_type import TaskType
from dataquality.utils.thread_pool import ThreadPoolManager
from dataquality.utils.vaex import validate_unique_ids
from tests.conftest import LOCATION
from fastai.vision.all import ImageDataLoaders, Resize, vision_learner, error_rate

from dataquality.clients.api import ApiClient
from tests.conftest import DEFAULT_PROJECT_ID, DEFAULT_RUN_ID


@patch.object(ApiClient, "valid_current_user", return_value=True)
@patch.object(dq.core.finish, "_version_check")
@patch.object(dq.core.finish, "_reset_run")
@patch.object(dq.core.finish, "upload_dq_log_file")
@patch.object(ApiClient, "make_request")
@patch.object(dq.core.finish, "wait_for_run")
@patch.object(ApiClient, "get_project_by_name")
@patch.object(ApiClient, "create_project")
@patch.object(ApiClient, "get_project_run_by_name", return_value={})
@patch.object(ApiClient, "create_run")
@patch("dataquality.core.init._check_dq_version")
@patch.object(dq.core.init.ApiClient, "valid_current_user", return_value=True)
def test_auto(
    mock_valid_user: MagicMock,
    mock_check_dq_version: MagicMock,
    mock_create_run: MagicMock,
    mock_get_project_run_by_name: MagicMock,
    mock_create_project: MagicMock,
    mock_get_project_by_name: MagicMock,
    set_test_config: Callable,
    mock_wait_for_run: MagicMock,
    mock_make_request: MagicMock,
    mock_upload_log_file: MagicMock,
    mock_reset_run: MagicMock,
    mock_version_check: MagicMock,
    cleanup_after_use: Generator,
) -> None:
    mock_get_project_by_name.return_value = {"id": DEFAULT_PROJECT_ID}
    mock_create_run.return_value = {"id": DEFAULT_RUN_ID}
    set_test_config(current_project_id=None, current_run_id=None)
    label_func = lambda x: x[0].isupper()  # noqa: E731
    image_files = list(map(Path, glob("tests/assets/images/*"))) * 10
    path = "tests/assets/images"
    dls = ImageDataLoaders.from_name_func(
        path,
        image_files,
        valid_pct=0.2,
        label_func=label_func,
        item_tfms=Resize(224),
        num_workers=1,
    )
    ThreadPoolManager.wait_for_threads()
    # validate_unique_ids(vaex.open(f"{LOCATION}/{split}/0/*.hdf5"), "epoch")
    # validate_unique_ids(vaex.open(f"{LOCATION}/{split}/1/*.hdf5"), "epoch")

    # dq.finish()
    learn = vision_learner(dls, "resnet34", metrics=error_rate)
    dqc = FastAiDQCallback(labels=["nocat", "cat"])
    learn.add_cb(dqc)
    learn.fine_tune(2)
