import os

from dataquality.core._config import set_config


def test_console_url() -> None:
    os.environ["GALILEO_CONSOLE_URL"] = "https://console.mytest.rungalileo.io"
    cfg = set_config()
    assert cfg.api_url == "https://api.mytest.rungalileo.io"
    assert cfg.minio_url == "data.mytest.rungalileo.io"
