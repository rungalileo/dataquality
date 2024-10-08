import os
from typing import Dict
from uuid import uuid4

import pytest

from dataquality import config
from dataquality.analytics import Analytics

os.environ["DQ_DEBUG"] = "1"


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


def test_mock_log_galileo_import():
    os.environ["DQ_TELEMETRICS"] = "1"
    a = Analytics(MockClient, config)
    a.last_log = {}
    a.log_import("test")
    assert a.last_log["value"] == "test", "No import detected"


def test_log_galileo_exception():
    os.environ["DQ_TELEMETRICS"] = "1"
    a = Analytics(MockClient, config)
    assert a._initialized, "Analytics not initialized"
    try:
        10 / 0
        a._log()
    except Exception as e:
        a.capture_exception(e)
        assert a.last_error["error_type"] == "ZeroDivisionError"
    with pytest.raises(Exception):
        10 / "1"


def test_log_galileo__import():
    os.environ["DQ_TELEMETRICS"] = "1"
    ac = Analytics(MockClient, config)
    config.api_url = "https://console.dev.rungalileo.io"
    ac.last_log = {}
    # Only check if telemetrics is enabled.
    if not ac._telemetrics_disabled:
        ac.config.current_project_id = uuid4()
        assert ac._initialized, "Analytics not initialized"
        ac._telemetrics_disabled = False
        ac.log_import("test")
        assert ac.last_log["value"] == "test", "No import detected"


def test_mock_log_galileo_import_disabled():
    os.environ["DQ_TELEMETRICS"] = "0"
    a_telemetrics_disabled = Analytics(MockClient, {"api_url": "https://customer"})
    a_telemetrics_disabled.last_log = {}
    a_telemetrics_disabled.log_import("test")
    log_result = a_telemetrics_disabled.last_log.get("value", "")
    assert log_result == "", "There should be no logging"
