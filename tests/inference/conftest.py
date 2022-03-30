import pytest

import dataquality


@pytest.fixture(autouse=True)
def reset_logger() -> None:
    # Only need to set one of data and models loggers since
    # they use the same logger config
    dataquality.get_data_logger().logger_config.reset()
