import os
from functools import partial
from typing import Callable
from unittest.mock import MagicMock, patch

import pytest

import dataquality
from dataquality import config
from dataquality.exceptions import GalileoException
from tests.utils.mock_request import (
    EXISTING_PROJECT,
    EXISTING_RUN,
    mocked_create_project_run,
    mocked_get_project_run,
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


def test_init_no_token_fails(set_test_config: Callable) -> None:
    set_test_config(token=None)
    with pytest.raises(GalileoException):
        # Raise error when no token is set
        dataquality.init(task_type="text_classification")


@patch("requests.get", side_effect=mocked_get_project_run)
@patch.object(dataquality.core.init.ApiClient, "valid_current_user", return_value=True)
def test_init_bad_task(
    mock_valid_user: MagicMock, mock_requests_get: MagicMock
) -> None:
    with pytest.raises(GalileoException):
        dataquality.init(task_type="not_text_classification")


@patch("builtins.input", return_value="")
@patch.object(dataquality.core.init.ApiClient, "get_current_user")
@patch.object(dataquality.core.init.ApiClient, "get_refresh_token")
def test_reconfigure_sets_env_vars(
    mock_get_refresh_token: MagicMock,
    mock_get_user: MagicMock,
    mock_input: MagicMock,
) -> None:
    # First token should fail, but second one should succeed
    mock_get_user.side_effect = [
        GalileoException,
        {"email": "me@example.com"},
    ]
    mock_get_refresh_token.side_effect = [
        {"access_token": "", "refresh_token": ""},
        {"access_token": "token", "refresh_token": "refresh_token"},
    ]

    os.environ["GALILEO_CONSOLE_URL"] = "https://console.fakecompany.io"
    os.environ["GALILEO_AUTH_CODE"] = ""
    dataquality.configure()
    assert dataquality.config.api_url == config.api_url == "https://api.fakecompany.io"
    assert dataquality.config.token == config.token == ""
    assert dataquality.config.refresh_token == config.refresh_token == ""

    os.environ["GALILEO_CONSOLE_URL"] = "https://console.newfake.de"
    os.environ["GALILEO_AUTH_CODE"] = "my-code"
    dataquality.configure()
    assert dataquality.config.api_url == config.api_url == "https://api.newfake.de"
    assert dataquality.config.token == config.token == "token"
    assert dataquality.config.refresh_token == config.refresh_token == "refresh_token"
    assert dataquality.config.current_user == config.current_user == "me@example.com"


@patch.object(dataquality.core.init.ApiClient, "get_current_user")
@patch.object(dataquality.core.init.ApiClient, "get_refresh_token")
def test_reconfigure_resets_user_token(
    mock_get_refresh_token: MagicMock,
    mock_get_user: MagicMock,
    set_test_config: Callable,
) -> None:
    mock_get_user.side_effect = [GalileoException]
    mock_get_refresh_token.side_effect = [
        {
            "access_token": "new_token",
            "refresh_token": "new_refresh_token",
        }
    ]
    os.environ["GALILEO_AUTH_CODE"] = "my-code"
    set_test_config(token="old_token")

    dataquality.configure()
    assert all(
        [
            config.token == "new_token",
            config.token != "old_token",
            config.refresh_token == "new_refresh_token",
        ]
    )


@patch.object(dataquality.core.init.ApiClient, "get_current_user")
@patch.object(dataquality.core.init.ApiClient, "get_refresh_token")
@patch.object(dataquality.core.init.ApiClient, "use_refresh_token")
def test_refresh_token_overrides_auth_code_on_login(
    mock_use_refresh_token: MagicMock,
    mock_get_refresh_token: MagicMock,
    mock_get_user: MagicMock,
) -> None:
    mock_get_user.side_effect = [
        {"email": "me@example.com"},
        {"email": "me@example.com"},
    ]
    mock_get_refresh_token.side_effect = [
        {
            "access_token": "first_token",
            "refresh_token": "first_refresh_token",
        }
    ]
    mock_use_refresh_token.side_effect = [
        {
            "access_token": "second_token",
            "refresh_token": "second_refresh_token",
        }
    ]

    os.environ["GALILEO_AUTH_CODE"] = "my-code"
    os.environ["GALILEO_CONSOLE_URL"] = "https://console.galileo.ai"

    dataquality.configure()
    assert dataquality.config.api_url == config.api_url == "https://api.galileo.ai"
    assert dataquality.config.token == config.token == "first_token"
    assert (
        dataquality.config.refresh_token
        == config.refresh_token
        == "first_refresh_token"
    )

    dataquality.login()
    assert dataquality.config.api_url == config.api_url == "https://api.galileo.ai"
    assert dataquality.config.token == config.token == "second_token"
    assert (
        dataquality.config.refresh_token
        == config.refresh_token
        == "second_refresh_token"
    )


@patch("requests.post", side_effect=mocked_create_project_run)
@patch("requests.get", side_effect=mocked_get_project_run)
@patch.object(dataquality.core.init.ApiClient, "valid_current_user", return_value=True)
@pytest.mark.parametrize(
    "run_name,exc",
    [
        ("my-run-#1", True),
        ("$run!$", True),
        ("a-bad-run##", True),
        ("A good run1", False),
        ("a-good-run-123", False),
        ("MY Run_1-2022", False),
    ],
)
def test_bad_names(
    mock_valid_user: MagicMock,
    mock_requests_get: MagicMock,
    mock_requests_post: MagicMock,
    run_name: str,
    exc: bool,
    set_test_config: Callable,
):
    init = partial(
        dataquality.init, task_type="text_classification", project_name="proj"
    )
    if exc:
        with pytest.raises(GalileoException):
            init(run_name=run_name)
    else:
        init(run_name=run_name)
