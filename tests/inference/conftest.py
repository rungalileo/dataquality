import pytest

import dataquality


@pytest.fixture(autouse=True)
def reset_loggers() -> None:
    dataquality.get_data_logger().logger_config.reset()
    dataquality.get_model_logger().logger_config.reset()
