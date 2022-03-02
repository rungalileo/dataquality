import os
from typing import Callable

import pytest

from dataquality.core._config import set_config, url_is_localhost


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
