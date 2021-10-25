import os
from unittest import mock

import pytest

import dataquality
from dataquality.core.auth import GALILEO_AUTH_METHOD
from tests.utils.mock_request import mocked_failed_requests_post, mocked_requests_post

config = dataquality.config


@mock.patch("requests.post", side_effect=mocked_requests_post)
def test_good_login(mock_post) -> None:
    os.environ[GALILEO_AUTH_METHOD] = "email"
    os.environ["GALILEO_USERNAME"] = "user"
    os.environ["GALILEO_PASSWORD"] = "password"
    dataquality.login()


@mock.patch("requests.post", side_effect=mocked_failed_requests_post)
def test_bad_login(mock_post) -> None:
    config.token = None
    os.environ[GALILEO_AUTH_METHOD] = "email"
    os.environ["GALILEO_USERNAME"] = "user"
    os.environ["GALILEO_PASSWORD"] = "password"
    with pytest.raises(Exception):
        dataquality.login()
