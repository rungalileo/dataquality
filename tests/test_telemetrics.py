from typing import Dict

import pytest

from dataquality.analytics import Analytics


class MockClient:
    def send_analytics(
        self,
        project_id: str = "UNKNOWN",
        run_id: str = "UNKNOWN",
        payload: Dict = {},
        run_task_type: str = "UNKNOWN",
        scope: str = "",
    ) -> None:
        pass


a = Analytics(MockClient, {})


def test_log_galileo_exception(set_test_config, cleanup_after_use):
    assert a._is_initialized, "Analytics not initialized"
    try:
        10 / 0
        a._log()
    except Exception as e:
        a.capture_exception(e)
        assert a.last_error["error_type"] == "ZeroDivisionError"
    with pytest.raises(Exception):
        10 / "1"
    # TODO


def test_log_galileo_import(set_test_config, cleanup_after_use):
    a.last_log = {}
    a.log_import("test")
    assert a.last_log["value"] == "test", "No import detected"


# def test_import_keys(set_test_config, cleanup_after_use):
#     modules = set(sys.modules)
#     assert "dataquality" not in modules
#     import dataquality as dq

#     modules = set(sys.modules)
#     assert "dataquality" in modules
#     dq.__version__
