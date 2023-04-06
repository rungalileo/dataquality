from glob import glob
from pathlib import Path
from typing import Any, Callable, Generator
from unittest.mock import MagicMock, patch

import pandas as pd
import torch
import torch.nn as nn
import vaex
from fastai.metrics import accuracy
from fastai.tabular.all import TabularDataLoaders, tabular_learner
from fastai.vision.all import ImageDataLoaders, Resize, error_rate, vision_learner

import dataquality as dq
from dataquality.clients.api import ApiClient
from dataquality.integrations.fastai import FastAiDQCallback, convert_img_dl_to_df
from dataquality.schemas.task_type import TaskType
from dataquality.utils.thread_pool import ThreadPoolManager
from dataquality.utils.vaex import validate_unique_ids
from tests.conftest import DEFAULT_PROJECT_ID, DEFAULT_RUN_ID, LOCATION


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
def test_auto(
    mock_valid_user: MagicMock,
    mock_dq_healthcheck: MagicMock,
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
        # bs=64,
        drop_last=False,
    )
    dq.init(task_type=TaskType.image_classification)
    dq.set_labels_for_run(["nocat", "cat"])
    for data, split in zip(dls, ["training", "validation"]):
        df = convert_img_dl_to_df(data)
        df["text"] = "s3://..."
        dq.log_image_dataset(df, split=split, imgs_location_colname="text")

    ThreadPoolManager.wait_for_threads()
    learn = vision_learner(dls, "resnet10t", metrics=error_rate)
    dqc = FastAiDQCallback(finish=False)
    learn.add_cb(dqc)
    learn.fine_tune(2, freeze_epochs=0)
    dq.log_image_dataset(df, imgs_location_colname="text", split="test")
    dl_test = learn.dls.test_dl(pd.Series(image_files[:-3]))
    dqc.prepare_split("test")
    preds, _ = learn.get_preds(dl=dl_test)
    for split in ["training", "validation"]:
        validate_unique_ids(vaex.open(f"{LOCATION}/{split}/1/*.hdf5"), "epoch")
    validate_unique_ids(vaex.open(f"{LOCATION}/test/0/*.hdf5"), "epoch")
    dqc.unwatch()
    dq.finish()


class PassThroughModel(nn.Module):
    embedding_dim: Any
    embedding: Any
    fc: Any

    def __init__(self, input_dim, embedding_dim, output_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.fc = nn.Linear(embedding_dim, output_dim)

    def forward(self, _x, x, *args, **kwargs):
        # print(x)
        # print(args)
        x = self.embedding(x.long())
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def init_weights(self):
        # Initialize the weights of the embedding layer to be an identity matrix
        self.embedding.weight.data.fill_(0)
        for i in range(self.embedding_dim):
            self.embedding.weight.data[i][0] = i
        # Initialize the bias of the fc layer to be zero
        self.fc.weight.data.fill_(1)
        self.fc.bias.data.fill_(0)


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
def test_tab(
    mock_valid_user: MagicMock,
    mock_dq_healthcheck: MagicMock,
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
    dq.init(task_type="image_classification")
    ds_len = 13
    df = pd.DataFrame(
        {
            "id": range(0, ds_len),
            "label": map(str, range(0, ds_len)),
            "text": range(0, ds_len),
        }
    )

    tdl = TabularDataLoaders.from_df(
        df.drop(["id"], axis=1),
        bs=12,
        cont_names=["text"],
        valid_idx=list(range(ds_len - int(ds_len * 0.3), ds_len)),
        y_names="label",
        drop_last=False,
    )
    tdl.device = torch.device("cpu")
    labels = list(map(str, range(0, ds_len)))
    dq.set_labels_for_run(labels)
    dq.log_dataset(df, split="training")
    dq.log_dataset(df, split="test")
    dq.log_dataset(df, split="validation")

    input_dim = ds_len
    embedding_dim = ds_len
    output_dim = 1
    model = PassThroughModel(input_dim, embedding_dim, output_dim)
    model.init_weights()
    model.cpu()

    def loss_fn(output, target):
        loss = nn.MSELoss()
        return loss(output, target)

    learn = tabular_learner(tdl, metrics=accuracy, loss_func=loss_fn)

    learn.model = model

    def empty_func():
        pass

    learn._backward = empty_func
    dqc = FastAiDQCallback(layer=model.fc, finish=False)
    learn.add_cb(dqc)
    learn.fit_one_cycle(1)
    print("done")
