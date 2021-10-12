import pytest

import dataquality
from dataquality.schemas.jsonl_logger import JsonlInputLogItem
from dataquality.schemas.split import Split
from dataquality.utils.thread_pool import ThreadPoolManager
from tests.utils.data_utils import NUM_LOGS, _log_data, validate_uploaded_data


def test_threaded_logging_and_upload(cleanup_after_use) -> None:
    """
    Tests that threaded calls to upload still yield non-missing datasets
    """
    num_records = 50
    _log_data(num_records=num_records)
    try:
        # Equivalent to the users `finish` call, but we don't want to clean up files yet
        ThreadPoolManager.wait_for_threads()
        validate_uploaded_data(num_records * NUM_LOGS)
    finally:
        ThreadPoolManager.wait_for_threads()
        dataquality._cleanup()


def test_set_data_version_fail():
    """
    You should not be able to set the data_schema_version of your logged data.
    An error should be thrown
    """
    with pytest.raises(ValueError):
        JsonlInputLogItem(
            id=1, split=Split.training, text="test", data_schema_version=5
        )
