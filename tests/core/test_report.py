from typing import Callable

from dataquality import Condition, get_data_logger, register_run_report


def test_register_run_report(set_test_config: Callable, condition: Condition) -> None:
    assert get_data_logger().logger_config.conditions == []
    assert get_data_logger().logger_config.report_emails == []

    register_run_report(conditions=[condition], emails=["foo@bar.com"])
    assert get_data_logger().logger_config.conditions == [condition]
    assert get_data_logger().logger_config.report_emails == ["foo@bar.com"]
