import os
from typing import Callable
from unittest.mock import MagicMock, patch

import pytest

import dataquality
from dataquality import config
from dataquality.core.auth import GALILEO_AUTH_METHOD
from dataquality.exceptions import GalileoException
from tests.exceptions import LoginInvoked
from tests.utils.mock_request import (
    EXISTING_PROJECT,
    EXISTING_RUN,
    mocked_create_project_run,
    mocked_get_project_run,
    mocked_login,
    mocked_login_requests,
    mocked_missing_project_run,
    mocked_missing_run,
)


@patch("requests.post", side_effect=mocked_create_project_run)
@patch("requests.get", side_effect=mocked_get_project_run)
@patch.object(dataquality.core.init.ApiClient, "valid_current_user", return_value=True)
def test_init(
    mock_valid_user: MagicMock,
    mock_requests_get: MagicMock,
    mock_requests_post: MagicMock,
    set_test_config: Callable,
) -> None:
    """Base case: Tests creating a new project and run"""
    dataquality.init(task_type="text_classification")
    assert config.current_run_id
    assert config.current_project_id


@patch("requests.post", side_effect=mocked_create_project_run)
@patch("requests.get", side_effect=mocked_get_project_run)
@patch.object(dataquality.core.init.ApiClient, "valid_current_user", return_value=True)
def test_init_reset_logger_config(
    mock_valid_user: MagicMock,
    mock_requests_get: MagicMock,
    mock_requests_post: MagicMock,
    set_test_config: Callable,
) -> None:
    """Base case: Tests creating a new project and run"""
    dataquality.init(task_type="text_classification")
    assert config.current_run_id
    assert config.current_project_id
    dataquality.set_labels_for_run(["a", "b", "c", "d"])
    dataquality.init(task_type="text_classification")
    assert not dataquality.get_data_logger().logger_config.labels


@patch("requests.post", side_effect=mocked_create_project_run)
@patch("requests.get", side_effect=mocked_get_project_run)
@patch.object(dataquality.core.init.ApiClient, "valid_current_user", return_value=True)
def test_init_private(
    mock_valid_user: MagicMock,
    mock_requests_get: MagicMock,
    mock_requests_post: MagicMock,
    set_test_config: Callable,
) -> None:
    """Base case: Tests creating a new project and run"""
    dataquality.init(task_type="text_classification", is_public=False)
    assert config.current_run_id
    assert config.current_project_id
    mock_create_project_call = mock_requests_post.call_args_list[0]
    assert mock_create_project_call.assert_called_with(is_public=False)


@patch("requests.post", side_effect=mocked_create_project_run)
@patch("requests.get", side_effect=mocked_get_project_run)
@patch.object(dataquality.core.init.ApiClient, "valid_current_user", return_value=True)
def test_init_existing_project(
    mock_valid_user: MagicMock,
    mock_requests_get: MagicMock,
    mock_requests_post: MagicMock,
    set_test_config: Callable,
) -> None:
    """Tests calling init passing in an existing project"""
    set_test_config(current_project_id=None, current_run_id=None)
    dataquality.init(task_type="text_classification", project_name=EXISTING_PROJECT)
    assert config.current_run_id
    assert config.current_project_id


@patch("requests.get", side_effect=mocked_missing_project_run)
@patch("requests.post", side_effect=mocked_create_project_run)
@patch.object(dataquality.core.init.ApiClient, "valid_current_user", return_value=True)
def test_init_new_project(
    mock_valid_user: MagicMock,
    mock_requests_get: MagicMock,
    mock_requests_post: MagicMock,
    set_test_config: Callable,
) -> None:
    """Tests calling init passing in a new project"""
    set_test_config(current_project_id=None, current_run_id=None)
    dataquality.init(task_type="text_classification", project_name="new_proj")
    assert config.current_run_id
    assert config.current_project_id


@patch("requests.get", side_effect=mocked_missing_run)
@patch("requests.post", side_effect=mocked_create_project_run)
@patch.object(dataquality.core.init.ApiClient, "valid_current_user", return_value=True)
def test_init_existing_project_new_run(
    mock_valid_user: MagicMock,
    mock_requests_get: MagicMock,
    mock_requests_post: MagicMock,
    set_test_config: Callable,
) -> None:
    """Tests calling init with an existing project but a new run"""
    set_test_config(current_project_id=None, current_run_id=None)
    dataquality.init(
        task_type="text_classification",
        project_name=EXISTING_PROJECT,
        run_name="new_run",
    )
    assert config.current_run_id
    assert config.current_project_id


@patch("requests.get", side_effect=mocked_get_project_run)
@patch("requests.post", side_effect=mocked_get_project_run)
@patch.object(dataquality.core.init.ApiClient, "valid_current_user", return_value=True)
def test_init_existing_project_run(
    mock_valid_user: MagicMock,
    mock_requests_get: MagicMock,
    mock_requests_post: MagicMock,
    set_test_config: Callable,
) -> None:
    """Tests calling init with an existing project and existing run"""
    set_test_config(current_project_id=None, current_run_id=None)
    dataquality.init(
        task_type="text_classification",
        project_name=EXISTING_PROJECT,
        run_name=EXISTING_RUN,
    )
    assert config.current_run_id
    assert config.current_project_id


@patch("requests.get", side_effect=mocked_missing_project_run)
@patch("requests.post", side_effect=mocked_create_project_run)
@patch.object(dataquality.core.init.ApiClient, "valid_current_user", return_value=True)
def test_init_new_project_run(
    mock_valid_user: MagicMock,
    mock_requests_get: MagicMock,
    mock_requests_post: MagicMock,
    set_test_config: Callable,
) -> None:
    """Tests calling init with a new project and new run"""
    set_test_config(current_project_id=None, current_run_id=None)
    dataquality.init(
        task_type="text_classification", project_name="new_proj", run_name="new_run"
    )
    assert config.current_run_id
    assert config.current_project_id


@patch("requests.get", side_effect=mocked_missing_project_run)
@patch.object(dataquality.core.init.ApiClient, "valid_current_user", return_value=True)
def test_init_only_run(
    mock_valid_user: MagicMock, mock_requests_get: MagicMock, set_test_config: Callable
) -> None:
    """Tests calling init only passing in a run"""
    set_test_config(current_project_id=None, current_run_id=None)
    dataquality.init(task_type="text_classification", run_name="a_run")
    assert not config.current_run_id
    assert not config.current_project_id


@patch("dataquality.core.init.login", side_effect=LoginInvoked)
def test_init_no_token_login(mock_login: MagicMock, set_test_config: Callable) -> None:
    set_test_config(token=None)
    with pytest.raises(LoginInvoked):
        # When no token is passed in we should call login
        dataquality.init(task_type="text_classification")
        mock_login.assert_called_once()


@patch("requests.post", side_effect=mocked_create_project_run)
@patch("requests.get", side_effect=mocked_get_project_run)
@patch("dataquality.core.init.login", side_effect=mocked_login)
def test_init_no_token_login_full(
    mock_login: MagicMock,
    mock_requests_get: MagicMock,
    mock_requests_post: MagicMock,
    set_test_config: Callable,
) -> None:
    set_test_config(token=None, current_project_id=None, current_run_id=None)
    # When no token is passed in we should call login
    dataquality.init(task_type="text_classification")
    mock_login.assert_called_once()
    # We also test the remaining init flow
    assert config.current_run_id
    assert config.current_project_id


@patch.object(
    dataquality.core.init.ApiClient, "get_current_user", side_effect=GalileoException
)
@patch("dataquality.core.init.login", side_effect=LoginInvoked)
def test_init_expired_token_login(
    mock_login: MagicMock, mock_current_user: MagicMock, set_test_config: Callable
) -> None:
    # When a token is passed in but user auth fails we should call login
    with pytest.raises(LoginInvoked):
        dataquality.init(task_type="text_classification")
        mock_login.assert_called_once()


@patch("requests.post", side_effect=mocked_create_project_run)
@patch("requests.get", side_effect=mocked_get_project_run)
@patch.object(
    dataquality.core.init.ApiClient, "get_current_user", side_effect=GalileoException
)
@patch("dataquality.core.init.login", side_effect=mocked_login)
def test_init_expired_token_login_full(
    mock_login: MagicMock,
    mock_current_user: MagicMock,
    mock_requests_get: MagicMock,
    mock_requests_post: MagicMock,
    set_test_config: Callable,
) -> None:
    set_test_config(current_project_id=None, current_run_id=None)
    # When a token is passed in but user auth fails we should call login
    dataquality.init(task_type="text_classification")
    mock_login.assert_called_once()
    # We also test the remaining init flow
    assert config.current_run_id
    assert config.current_project_id


@patch.object(dataquality.core.init.ApiClient, "valid_current_user", return_value=False)
@patch("dataquality.core.init.login", side_effect=LoginInvoked)
def test_init_invalid_user_login(
    mock_login: MagicMock, mock_valid_user: MagicMock
) -> None:
    # When current user is not valid we should call login
    with pytest.raises(LoginInvoked):
        dataquality.init(task_type="text_classification")
        mock_login.assert_called_once()


@patch("requests.post", side_effect=mocked_create_project_run)
@patch("requests.get", side_effect=mocked_get_project_run)
@patch.object(dataquality.core.init.ApiClient, "valid_current_user", return_value=False)
@patch("dataquality.core.init.login", side_effect=mocked_login)
def test_init_invalid_user_login_full(
    mock_login: MagicMock,
    mock_valid_user: MagicMock,
    mock_requests_get: MagicMock,
    mock_requests_post: MagicMock,
    set_test_config: Callable,
) -> None:
    set_test_config(current_project_id=None, current_run_id=None)
    # When current user is not valid we should call login
    dataquality.init(task_type="text_classification")
    mock_login.assert_called_once()
    # We also test the remaining init flow
    assert config.current_run_id
    assert config.current_project_id


@patch("requests.get", side_effect=mocked_get_project_run)
@patch.object(dataquality.core.init.ApiClient, "valid_current_user", return_value=True)
def test_init_bad_task(
    mock_valid_user: MagicMock, mock_requests_get: MagicMock
) -> None:
    with pytest.raises(GalileoException):
        dataquality.init(task_type="not_text_classification")


@patch("dataquality.login")
def test_reconfigure_sets_env_vars(mock_login: MagicMock) -> None:
    os.environ["GALILEO_CONSOLE_URL"] = "https://console.fakecompany.io"
    dataquality.configure()
    assert dataquality.config.minio_url == config.minio_url == "data.fakecompany.io"
    assert dataquality.config.api_url == config.api_url == "https://api.fakecompany.io"
    os.environ["GALILEO_CONSOLE_URL"] = "https://console.newfake.de"
    dataquality.configure()
    assert dataquality.config.minio_url == config.minio_url == "data.newfake.de"
    assert dataquality.config.api_url == config.api_url == "https://api.newfake.de"

    assert mock_login.call_count == 2


@patch("requests.post", side_effect=mocked_login_requests)
@patch("requests.get", side_effect=mocked_login_requests)
def test_reconfigure_resets_user_token(
    mock_get_request: MagicMock,
    mock_post_request: MagicMock,
    set_test_config: Callable,
) -> None:
    set_test_config(token="old_token")

    os.environ[GALILEO_AUTH_METHOD] = "email"
    os.environ["GALILEO_USERNAME"] = "user"
    os.environ["GALILEO_PASSWORD"] = "password"
    dataquality.configure()
    assert all([config.token == "mock_token", config.token != "old_token"])


@patch("dataquality.login", side_effect=mocked_login)
def test_reconfigure_resets_user_token_login_mocked(
    mock_login: MagicMock, set_test_config: Callable
) -> None:
    set_test_config(token="old_token")
    dataquality.configure()
    assert all([config.token == "mock_token", config.token != "old_token"])
    mock_login.assert_called_once()
