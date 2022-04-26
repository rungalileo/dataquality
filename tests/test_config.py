import os
from typing import Callable
from unittest.mock import MagicMock, patch

import pytest
import requests

from dataquality.core._config import set_config, url_is_localhost
from dataquality.exceptions import GalileoException


def test_console_url(set_test_config: Callable) -> None:
    os.environ["GALILEO_CONSOLE_URL"] = "https://console.mytest.rungalileo.io"
    cfg = set_config()
    assert cfg.api_url == "https://api.mytest.rungalileo.io"
    assert cfg.minio_url == "data.mytest.rungalileo.io"


@pytest.mark.parametrize(
    "console_url",
    ["http://localhost:3000", "http://127.0.0.1:3000"],
)
def test_console_url_local(set_test_config: Callable, console_url: str) -> None:
    os.environ["GALILEO_CONSOLE_URL"] = console_url
    cfg = set_config()
    assert cfg.api_url == "http://localhost:8088"
    assert cfg.minio_url == "localhost:9000"


def test_bad_console_url(set_test_config: Callable) -> None:
    """If console is not in the console url, dont use it"""
    os.environ["GALILEO_CONSOLE_URL"] = "https://something.mytest2.rungalileo.io"
    cfg = set_config()
    assert cfg.api_url != "https://api.mytest2.rungalileo.io"


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://console.domain.com", False),
        ("http://localhost:3000", True),
        ("http://127.0.0.1:8088", True),
    ],
)
def test_url_is_localhost(url: str, expected: bool) -> bool:
    assert url_is_localhost(url) is expected


@patch("dataquality.core._config.requests.get")
@pytest.mark.noautofixt
def test_validate_bad_config_url(get_mock: MagicMock) -> None:
    get_mock.side_effect = requests.ConnectionError("No connection!")
    bad_url = "https://console.mytest2.rungalileo.io/badurl"
    os.environ["GALILEO_CONSOLE_URL"] = bad_url
    with pytest.raises(GalileoException) as e:
        set_config()
    assert e.value.args[0].startswith(f"The provided console URL {bad_url} is invalid")


def test_handle_extra_slash_in_console(set_test_config: Callable) -> None:
    """Removes slashes in an otherwise valid console url"""
    os.environ["GALILEO_CONSOLE_URL"] = "https://console.mytest2.rungalileo.io///"
    cfg = set_config()
    assert cfg.api_url == "https://api.mytest2.rungalileo.io"
