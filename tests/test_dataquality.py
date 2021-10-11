import dataquality
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
