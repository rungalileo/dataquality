import os
from unittest import mock
from unittest.mock import MagicMock

import pytest

import dataquality
from dataquality.core.auth import GALILEO_AUTH_METHOD
from dataquality.exceptions import GalileoException
from tests.utils.mock_request import mocked_failed_login_requests, mocked_login_requests

config = dataquality.config


@mock.patch("requests.post", side_effect=mocked_login_requests)
@mock.patch("requests.get", side_effect=mocked_login_requests)
def test_good_login(
    mock_get_current_user: MagicMock,
    mock_login: MagicMock,
) -> None:
    os.environ[GALILEO_AUTH_METHOD] = "email"
    os.environ["GALILEO_USERNAME"] = "user"
    os.environ["GALILEO_PASSWORD"] = "password"
    config.token = "mytoken"
    dataquality.login()


@mock.patch("requests.post", side_effect=mocked_failed_login_requests)
def test_bad_login(mock_post: MagicMock) -> None:
    tok = config.token
    config.token = None
    os.environ[GALILEO_AUTH_METHOD] = "email"
    os.environ["GALILEO_USERNAME"] = "user"
    os.environ["GALILEO_PASSWORD"] = "password"
    with pytest.raises(GalileoException):
        dataquality.login()
    config.token = tok
