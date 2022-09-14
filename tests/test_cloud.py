import pytest

from dataquality.core._config import config
from dataquality.utils.cloud import is_galileo_cloud

CLOUD_URL = "https://console.cloud.rungalileo.io"


@pytest.mark.parametrize(
    "api_url, expected",
    [
        ("https://console.cloud.rungalileo.io", True),
        ("https://console.fake.rungalileo.io", False),
    ],
)
def test_is_galileo_cloud(api_url: str, expected: bool) -> None:
    config.api_url = api_url
    assert is_galileo_cloud() is expected
