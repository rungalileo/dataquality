from dataquality.core._config import CLOUD_URL, config


def is_galileo_cloud() -> bool:
    return config.api_url == CLOUD_URL.replace("console", "api")
