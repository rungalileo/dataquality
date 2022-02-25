from unittest import mock
from unittest.mock import MagicMock

import pytest

import dataquality


def test_finish_no_init() -> None:
    """
    Tests finish without an init call
    """
    dataquality.config.current_run_id = dataquality.config.current_project_id = None
    with pytest.raises(AssertionError):
        dataquality.finish()


@mock.patch.object(dataquality.core.init.ApiClient, "wait_for_run")
def test_wait_for_run(mock_client: MagicMock) -> None:
    """
    Tests that wait_for_run calls ApiClient
    """
    dataquality.wait_for_run(project_name="Carrots", run_name="Rhubarb")
    mock_client.assert_called_once_with(project_name="Carrots", run_name="Rhubarb")


@mock.patch.object(
    dataquality.core.init.ApiClient,
    "get_run_status",
    return_value={"status": "started"},
)
def test_get_run_status(mock_client: MagicMock) -> None:
    """
    Tests that get_run_status calls ApiClient
    """
    status = dataquality.get_run_status(project_name="Carrots", run_name="Rhubarb")
    mock_client.assert_called_once_with(project_name="Carrots", run_name="Rhubarb")
    assert status.get("status") == "started"
