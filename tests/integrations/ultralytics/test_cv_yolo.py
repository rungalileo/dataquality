from glob import glob
from pathlib import Path
from typing import Any, Callable, Generator
from unittest.mock import MagicMock, patch

import pandas as pd
import torch
import torch.nn as nn
from fastai.metrics import accuracy
from fastai.tabular.all import TabularDataLoaders, tabular_learner
from fastai.vision.all import ImageDataLoaders, Resize, error_rate, vision_learner

import dataquality as dq
from dataquality.clients.api import ApiClient
from dataquality.integrations.fastai import FastAiDQCallback, convert_img_dl_to_df
from dataquality.integrations.ultralytics import Callback, add_callback
from dataquality.utils.thread_pool import ThreadPoolManager
from dataquality.utils.ultralytics import non_max_suppression
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
    from ultralytics import YOLO

    model = YOLO('yolov8n.pt')  
    cb = Callback(nms_fn=non_max_suppression)
    add_callback(model, cb)
    preds = model.val(data="coco.yaml")

    df = pd.DataFrame(
        cb.validator.dataloader.dataset.get_labels()
    )

    #df_lookup = df.reset_index().set_index("im_file")["index"].to_dict()

    ThreadPoolManager.wait_for_threads()

    # validate_unique_ids(vaex.open(f"{LOCATION}/{split}/0/*.hdf5"), "epoch")
    # validate_unique_ids(vaex.open(f"{LOCATION}/{split}/1/*.hdf5"), "epoch")

    # dq.finish()
