from datetime import datetime
from typing import List, Optional
from uuid import UUID

from vaex.dataframe import DataFrame

from dataquality.clients.api import ApiClient
from dataquality.core.log import get_data_logger
from dataquality.metrics import get_dataframe
from dataquality.schemas.condition import Condition
from dataquality.schemas.report import (
    ConditionStatus,
    ReportConditionData,
    RunReportData,
    SplitConditionData,
)
from dataquality.schemas.split import Split

api_client = ApiClient()


def register_run_report(conditions: List[Condition], emails: List[str]) -> None:
    """Register conditions and emails for a run report.

    After a run is finished, a report will be sent to the specified emails.
    """
    get_data_logger().logger_config.conditions = conditions
    get_data_logger().logger_config.report_emails = emails


def _get_email_datetime() -> str:
    """Get the current datetime in a human readable format."""
    return datetime.now().strftime("%A %m/%d/%Y, %H:%M:%S")


def _condition_to_verbose_string(condition: Condition) -> str:
    """Convert a condition to a verbose string."""
    return (
        f"{condition.agg} {condition.metric} {condition.operator} {condition.threshold}"
    )


def _get_report_results_for_split(
    condition: Condition,
    df: DataFrame,
    split: Split,
    inference_name: Optional[str] = None,
) -> SplitConditionData:
    passes, val = condition.evaluate(df)
    return SplitConditionData(
        split=split.value,
        inference_name=inference_name,
        status=ConditionStatus.passed if passes else ConditionStatus.failed,
        ground_truth=round(val, 3),
        link=None,  # TODO: add deep link, v2 of reports
    )


def _get_report_results_for_condition(
    condition: Condition,
    splits: List[str],
    inference_names: List[str],
    project_name: str,
    run_name: str,
) -> ReportConditionData:
    """Get the results for a condition."""
    split_data = []
    for split in splits:
        split = Split[split]
        if split == Split.inference:
            for inf_name in inference_names:
                df = get_dataframe(project_name, run_name, split, inf_name)
                split_data.append(
                    _get_report_results_for_split(condition, df, split, inf_name)
                )
        else:
            df = get_dataframe(project_name, run_name, split, as_pandas=False)
            split_data.append(_get_report_results_for_split(condition, df, split))

    return ReportConditionData(
        condition=condition.metric,
        criteria=_condition_to_verbose_string(condition),
        splits=split_data,
    )


def build_run_report(
    conditions: List[Condition],
    emails: List[str],
    project_id: UUID,
    run_id: UUID,
    link: str,
) -> None:
    """Build a run report and send it to the specified emails."""
    project_name = api_client.get_project(project_id)["name"]
    run_name = api_client.get_project_run(project_id, run_id)["name"]
    logged_splits = api_client.get_splits(project_id, run_id)["splits"]
    inference_names = api_client.get_inference_names(project_id, run_id)[
        "inference_names"
    ]

    report_data = RunReportData(
        email_subject=f"Run Report: {run_name}",
        template="run_report",
        project_name=project_name,
        run_name=run_name,
        created_at=_get_email_datetime(),
        link=link,
        conditions=[],
    )

    for c in conditions:
        report_data.conditions.append(
            _get_report_results_for_condition(
                c, logged_splits, inference_names, project_name, run_name
            )
        )

    api_client.notify_email(report_data.dict(), "run_report", emails)
