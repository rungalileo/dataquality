import os
from typing import Callable

from dataquality.core._config import set_config


def test_console_url(set_test_config: Callable) -> None:
    os.environ["GALILEO_CONSOLE_URL"] = "https://console.mytest.rungalileo.io"
    cfg = set_config()
    assert cfg.api_url == "https://api.mytest.rungalileo.io"
    assert cfg.minio_url == "data.mytest.rungalileo.io"


def test_bad_console_url(set_test_config: Callable) -> None:
    """If console is not in the console url, dont use it"""
    os.environ["GALILEO_CONSOLE_URL"] = "https://something.mytest2.rungalileo.io"
    cfg = set_config()
    assert cfg.api_url != "https://api.mytest2.rungalileo.io"
