from typing import Callable
from unittest import mock
from uuid import uuid4

import freezegun
import vaex

from dataquality import (
    AggregateFunction,
    ConditionFilter,
    Operator,
    build_run_report,
    get_data_logger,
    register_run_report,
)
from dataquality.clients.api import ApiClient
from dataquality.core.report import (
    _condition_to_verbose_string,
    _condition_valid_for_df,
    _get_email_datetime,
)
from dataquality.schemas.report import ConditionStatus

# https://stackoverflow.com/questions/73007409/freezeguns-freeze-time-throws-odd-transformers-error-when-used
freezegun.configure(extend_ignore_list=["transformers"])


def test_register_run_report(
    set_test_config: Callable, test_condition: Callable
) -> None:
    condition = test_condition()
    assert get_data_logger().logger_config.conditions == []
    assert get_data_logger().logger_config.report_emails == []

    register_run_report(conditions=[condition], emails=["foo@bar.com"])
    assert get_data_logger().logger_config.conditions == [condition]
    assert get_data_logger().logger_config.report_emails == ["foo@bar.com"]


@freezegun.freeze_time("2021-09-14 15:00:00")
def test_get_email_datetime() -> None:
    assert _get_email_datetime() == "Tuesday 09/14/2021, 15:00:00"


def test_condition_to_verbose_string(test_condition: Callable) -> None:
    condition = test_condition()
    assert (
        _condition_to_verbose_string(condition) == "Average confidence is less than 0.5"
    )


def test_condition_valid_for_df(test_condition: Callable) -> None:
    """Since metric 'is_drifted' isn't present in DF, condition should be invalid."""
    condition = test_condition(  # pct drifted > 0.5 ?
        agg=AggregateFunction.pct,
        metric=None,
        operator=Operator.gte,
        threshold=0.5,
        filters=[
            ConditionFilter(
                metric="is_drifted",
                operator=Operator.eq,
                value=True,
            )
        ],
    )
    inp = dict(
        id=range(0, 10),
        confidence=[0.1] * 10,
    )  # True avg confidence is 0.1
    df = vaex.from_dict(inp)

    assert _condition_valid_for_df(df, condition) is False


@freezegun.freeze_time("2021-09-14 15:00:00")
@mock.patch("dataquality.core.report.get_dataframe")
@mock.patch.object(ApiClient, "notify_email")
@mock.patch.object(ApiClient, "get_inference_names")
@mock.patch.object(ApiClient, "get_splits")
@mock.patch.object(ApiClient, "get_project_run")
@mock.patch.object(ApiClient, "get_project")
def test_build_run_report_e2e(
    mock_get_project: mock.MagicMock,
    mock_get_project_run: mock.MagicMock,
    mock_get_splits: mock.MagicMock,
    mock_get_inference_names: mock.MagicMock,
    mock_notify_email: mock.MagicMock,
    mock_get_dataframe: mock.MagicMock,
    test_condition: Callable,
) -> None:
    # Set up mocks
    mock_get_project.return_value = {"name": "test_project"}
    mock_get_project_run.return_value = {"name": "test_run"}
    mock_get_splits.return_value = {"splits": ["training", "inference"]}
    mock_get_inference_names.return_value = {"inference_names": ["inf1", "inf2"]}

    # Set up dataframes
    inp = dict(
        id=range(0, 10),
        confidence=[0.1] * 10,
    )  # True avg confidence is 0.1
    train_df = vaex.from_dict(inp)
    inf_inp = dict(
        id=range(0, 10),
        confidence=[0.4] * 10,
        is_drifted=[True] * 7 + [False] * 3,
    )  # True avg confidence is 0.4
    inf_df1 = vaex.from_dict(inf_inp)
    inf_inp2 = dict(
        id=range(0, 10),
        confidence=[0.7] * 10,
        is_drifted=[False] * 7 + [True] * 3,
    )  # True avg confidence is 0.7
    inf_df2 = vaex.from_dict(inf_inp2)
    mock_get_dataframe.side_effect = [train_df, inf_df1, inf_df2]

    # Create conditions and run report
    condition1 = test_condition(  # avg confidence < 0.4 ?
        agg=AggregateFunction.avg,
        metric="confidence",
        operator=Operator.lt,
        threshold=0.4,
    )
    condition2 = test_condition(  # pct drifted > 0.5 ?
        agg=AggregateFunction.pct,
        metric=None,
        operator=Operator.gte,
        threshold=0.5,
        filters=[
            ConditionFilter(
                metric="is_drifted",
                operator=Operator.eq,
                value=True,
            )
        ],
    )
    build_run_report(
        conditions=[condition1, condition2],
        emails=["foo@bar.com"],
        project_id=uuid4(),
        run_id=uuid4(),
        link="www.foobar.com",
    )

    expected_report_data = {
        "email_subject": "Run Report: test_run",
        "project_name": "test_project",
        "run_name": "test_run",
        "created_at": "Tuesday 09/14/2021, 15:00:00",
        "link": "www.foobar.com",
        "conditions": [
            {
                "metric": "confidence",
                "condition": "Average confidence is less than 0.4",
                "splits": [
                    {
                        "split": "training",
                        "inference_name": "",
                        "status": ConditionStatus.passed,
                        "link": None,
                        "ground_truth": 0.1,
                    },
                    {
                        "split": "inference",
                        "inference_name": "inf1",
                        "status": ConditionStatus.passed,
                        "link": None,
                        "ground_truth": 0.4,
                    },
                    {
                        "split": "inference",
                        "inference_name": "inf2",
                        "status": ConditionStatus.failed,
                        "link": None,
                        "ground_truth": 0.7,
                    },
                ],
            },
            {
                "metric": "is_drifted",
                "condition": (
                    "Percentage (is_drifted is equal to 1.0) is greater than or "
                    "equal to 0.5"
                ),
                "splits": [
                    {
                        "split": "inference",
                        "inference_name": "inf1",
                        "status": ConditionStatus.passed,
                        "link": None,
                        "ground_truth": 0.7,
                    },
                    {
                        "split": "inference",
                        "inference_name": "inf2",
                        "status": ConditionStatus.failed,
                        "link": None,
                        "ground_truth": 0.3,
                    },
                ],
            },
        ],
    }
    mock_notify_email.assert_called_once_with(
        expected_report_data, "run_report", ["foo@bar.com"]
    )
    # Assert that caching prevented all 6 calls to get_dataframes
    assert mock_get_dataframe.call_count == 3
