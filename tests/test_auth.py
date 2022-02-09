import os
from unittest import mock

import pytest

import dataquality
from dataquality.core.auth import GALILEO_AUTH_METHOD
from tests.utils.mock_request import mocked_failed_login_requests, mocked_login_requests

config = dataquality.config


@mock.patch("requests.post", side_effect=mocked_login_requests)
@mock.patch("requests.get", side_effect=mocked_login_requests)
def test_good_login(manage_token, *args) -> None:
    os.environ[GALILEO_AUTH_METHOD] = "email"
    os.environ["GALILEO_USERNAME"] = "user"
    os.environ["GALILEO_PASSWORD"] = "password"
    dataquality.login()


@mock.patch("requests.post", side_effect=mocked_failed_login_requests)
def test_bad_login(mock_post, manage_token) -> None:
    config.token = None
    os.environ[GALILEO_AUTH_METHOD] = "email"
    os.environ["GALILEO_USERNAME"] = "user"
    os.environ["GALILEO_PASSWORD"] = "password"
    with pytest.raises(Exception):
        dataquality.login()
