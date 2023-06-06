from dataquality.core._config import CLOUD_URL
from dataquality.core._config import config


def is_galileo_cloud() -> bool:
    return config.api_url == CLOUD_URL.replace("console", "api")
