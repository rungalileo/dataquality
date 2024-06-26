from typing import Callable
from unittest import mock
from unittest.mock import MagicMock

import numpy as np

import dataquality as dq
from dataquality.utils.dq_logger import dq_log_file_path


@mock.patch("dataquality.core.finish._reset_run")
@mock.patch.object(dq.clients.api.ApiClient, "make_request")
@mock.patch.object(dq.clients.api.ApiClient, "get_presigned_url")
@mock.patch.object(
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
@mock.patch.object(dq.clients.objectstore.ObjectStore, "download_file")
@mock.patch.object(dq.clients.objectstore.ObjectStore, "_upload_file_from_local")
def test_std_log(
    mock_upload_from_local: MagicMock,
    mock_download: MagicMock,
    mock_bucket_names: MagicMock,
    mock_presigned_url: MagicMock,
    mock_finish: MagicMock,
    mock_reset_run: MagicMock,
    set_test_config: Callable,
    cleanup_after_use: Callable,
    input_data: Callable,
) -> None:
    """Validate that we interrupt the main process if issues occur while logging"""
    dq.set_labels_for_run(["APPLE", "ORANGE"])
    training_data = input_data(meta={"training_meta": [1.414, 123]})
    dq.log_data_samples(**training_data)

    dq.set_split("training")
    dq.log_model_outputs(
        embs=np.random.rand(2, 100),
        logits=np.random.rand(2, 2),
        ids=[1, 2],
        epoch=0,
    )
    mock_finish.return_value = {"job_name": "test_job", "link": "link"}
    mock_presigned_url.return_value = "https://google.com"
    mock_upload_from_local.return_value = None
    dq.finish(wait=False)
    mock_download.return_value = "my-file"

    x = dq.get_dq_log_file()
    assert x == dq_log_file_path()
