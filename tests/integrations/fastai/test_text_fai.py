from typing import Callable, Generator
from unittest.mock import MagicMock, patch

import pandas as pd
import torch
import vaex
from fastai.metrics import accuracy
from fastai.text.all import TextDataLoaders
from fastai.text.learner import text_classifier_learner
from fastai.text.models.awdlstm import AWD_LSTM

import dataquality as dq
from dataquality.clients.api import ApiClient
from dataquality.integrations.fastai import FastAiDQCallback, extract_split_indices
from dataquality.schemas.split import Split
from dataquality.schemas.task_type import TaskType
from dataquality.utils.thread_pool import ThreadPoolManager
from dataquality.utils.vaex import validate_unique_ids
from tests.conftest import DEFAULT_PROJECT_ID, DEFAULT_RUN_ID, LOCATION
from tests.test_utils.mock_data import mock_dict

df = pd.DataFrame(mock_dict)
df["is_valid"] = False * (len(df) / 2) + True * (len(df) / 2)


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
@patch.object(
    dq.clients.api.ApiClient,
    "get_healthcheck_dq",
    return_value={
        "bucket_names": {
            "images": "galileo-images",
            "results": "galileo-project-runs-results",
            "root": "galileo-project-runs",
        },
        "minio_fqdn": "127.0.0.1:9000",
    },
)
@patch.object(dq.core.init.ApiClient, "valid_current_user", return_value=True)
def test_end2end_fai(
    mock_dq_healthcheck: MagicMock,
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
    global df
    mock_get_project_by_name.return_value = {"id": DEFAULT_PROJECT_ID}
    mock_create_run.return_value = {"id": DEFAULT_RUN_ID}
    set_test_config(current_project_id=None, current_run_id=None)
    dls = TextDataLoaders.from_df(
        df, text_col="text", label_col="label", drop_last=False, bs=2
    )
    dq.init(task_type=TaskType.text_classification)
    labels = dls.vocab[-1]
    dq.set_labels_for_run(list(labels))
    train_ids, valid_ids = extract_split_indices(dls)
    df["id"] = df.index
    for test_split, split_idx in zip(
        [Split.training, Split.validation], [train_ids, valid_ids]
    ):
        df_split = df.iloc[split_idx]
        dq.log_dataset(df_split, split=test_split)

    ThreadPoolManager.wait_for_threads()
    dqc = FastAiDQCallback(finish=False)
    learn = text_classifier_learner(dls, AWD_LSTM, drop_mult=0.5, metrics=accuracy)
    dls.device = torch.device("cpu")
    learn.add_cb(dqc)
    learn.fine_tune(1, 1e-2, freeze_epochs=0)
    for test_split in ["training", "validation"]:
        validate_unique_ids(vaex.open(f"{LOCATION}/{test_split}/0/*.hdf5"), "epoch")
    dqc.unwatch()
    dq.finish()
