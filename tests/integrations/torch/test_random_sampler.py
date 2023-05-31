from typing import Callable, Generator, Tuple
from unittest.mock import MagicMock, patch

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

import dataquality as dq
from dataquality.clients.api import ApiClient
from dataquality.schemas.split import Split
from tests.conftest import DEFAULT_PROJECT_ID, DEFAULT_RUN_ID


# Assuming your labels are the target for your model
class TextDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        id_column: str,
        text_column: str,
        label_column: str,
    ) -> None:
        self.dataframe = dataframe
        self.text = dataframe[text_column]
        self.ids = dataframe[id_column]
        self.labels = dataframe[label_column]
        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(self.labels)

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> Tuple:
        text = self.text[idx]
        label = self.labels[idx]
        ids = self.ids[idx]
        return ids, text, label


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
def test_random(
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
    labels = ["a", "b"]
    dq.set_labels_for_run(labels)

    dq.log_data_samples(
        texts=["a", "b", "a"],
        labels=["a", "b", "a"],
        ids=[0, 1, 2],
        split=Split.training,
    )
    dq.log_model_outputs(
        embs=[[0, 0], [1, 1]],
        ids=[0, 1],
        probs=[[0, 1], [1, 0]],
        split=Split.training,
        epoch=0,
    )
    dq.finish()
