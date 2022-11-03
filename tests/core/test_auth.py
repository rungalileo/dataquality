import os
from typing import Callable
from unittest import mock
from unittest.mock import MagicMock

import pytest

import dataquality
from dataquality.core.auth import GALILEO_AUTH_METHOD
from dataquality.exceptions import GalileoException
from tests.test_utils.mock_request import (
    mocked_failed_login_requests,
    mocked_login_requests,
)

config = dataquality.config


@mock.patch("requests.post", side_effect=mocked_login_requests)
@mock.patch("requests.get", side_effect=mocked_login_requests)
def test_good_login(
    mock_get_current_user: MagicMock, mock_login: MagicMock, set_test_config: Callable
) -> None:
    os.environ[GALILEO_AUTH_METHOD] = "email"
    os.environ["GALILEO_USERNAME"] = "user"
    os.environ["GALILEO_PASSWORD"] = "password"
    set_test_config(token="mytoken")
    dataquality.login()


@mock.patch("requests.post", side_effect=mocked_failed_login_requests)
def test_bad_login(mock_post: MagicMock, set_test_config: Callable) -> None:
    set_test_config(token=None)
    os.environ[GALILEO_AUTH_METHOD] = "email"
    os.environ["GALILEO_USERNAME"] = "user"
    os.environ["GALILEO_PASSWORD"] = "password"
    with pytest.raises(GalileoException) as e:
        dataquality.login()
    assert e.value.args[0] == (
        "Issue authenticating: Incorrect login credentials. If you need to reset your "
        "password, go to http://localhost:8088/forgot-password"
    )
