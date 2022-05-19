from typing import Callable
from unittest import mock
from unittest.mock import MagicMock

import numpy as np

import dataquality as dq
from dataquality.utils.dq_logger import dq_log_file_path


@mock.patch("dataquality.core.finish._reset_run")
@mock.patch("dataquality.core.finish._version_check")
@mock.patch.object(dq.clients.api.ApiClient, "make_request")
@mock.patch.object(dq.clients.objectstore.Minio, "fget_object")
@mock.patch.object(dq.clients.objectstore.Minio, "fput_object")
def test_std_log(
    mock_put_object: MagicMock,
    mock_get_object: MagicMock,
    mock_finish: MagicMock,
    mock_version_check: MagicMock,
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
    dq.finish()

    mock_put_object.assert_called_once_with(
        "galileo-project-runs",
        f"{dq.config.current_project_id}/{dq.config.current_run_id}/out/out.log",
        file_path=dq_log_file_path(),
        content_type="text/plain",
    )
    x = dq.get_dq_log_file()
    assert x == dq_log_file_path()
    mock_get_object.assert_called_once_with(
        "galileo-project-runs",
        f"{dq.config.current_project_id}/{dq.config.current_run_id}/out/out.log",
        dq_log_file_path(),
    )
