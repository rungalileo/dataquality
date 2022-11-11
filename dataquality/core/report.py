from datetime import datetime
from typing import Dict, List, Tuple
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


def _get_metric(condition: Condition) -> str:
    """Get the metric name for a condition."""
    if condition.metric:
        return condition.metric
    elif condition.filters:
        return condition.filters[0].metric
    else:
        return ""


def _condition_to_verbose_string(condition: Condition) -> str:
    """Convert a condition to a verbose string."""
    metric_str = condition.metric
    if not metric_str and condition.filters:
        f = condition.filters[0]
        metric_str = f"({f.metric} {f.operator} {f.value})"

    return f"{condition.agg} {metric_str} {condition.operator} {condition.threshold}"


def _get_dataframe_from_cache_or_server(
    project_name: str,
    run_name: str,
    split: Split,
    inference_name: str = "",
    split_cache: Dict[Tuple, DataFrame] = None,
) -> DataFrame:
    """Get a dataframe from the cache or the server."""
    if split_cache is None:
        split_cache = {}

    if (project_name, run_name, split, inference_name) in split_cache:
        return split_cache[(project_name, run_name, split, inference_name)]

    df = get_dataframe(project_name, run_name, split, inference_name, as_pandas=False)
    split_cache[(project_name, run_name, split, inference_name)] = df
    return df


def _condition_valid_for_df(df: DataFrame, condition: Condition) -> bool:
    """Check if a condition is valid for a dataframe.

    For example, if condition is related to 'is_drifted' but the dataframe
    is a training df, then the condition is not valid and we should skip it.
    """
    df_cols = df.get_column_names()
    metric_cols = set(
        [condition.metric] + [filt.metric for filt in condition.filters or []]
    )
    metric_cols.discard(None)  # Remove 'None' if it exists

    for col in metric_cols:
        if col not in df_cols:
            return False

    return True


def _add_split_data_for_condition(
    condition: Condition,
    project_name: str,
    run_name: str,
    split: Split,
    split_data: List[SplitConditionData],
    split_cache: Dict[Tuple, DataFrame] = None,
    inference_name: str = "",
) -> List[SplitConditionData]:
    df = _get_dataframe_from_cache_or_server(
        project_name, run_name, split, inference_name, split_cache
    )
    if not _condition_valid_for_df(df, condition):
        return split_data

    passes, val = condition.evaluate(df)
    split_data.append(
        SplitConditionData(
            split=split.value,
            inference_name=inference_name,
            status=ConditionStatus.passed if passes else ConditionStatus.failed,
            ground_truth=round(val, 3),
            link=None,  # TODO: add deep link, v2 of reports
        )
    )
    return split_data


def _get_report_results_for_condition(
    condition: Condition,
    splits: List[str],
    inference_names: List[str],
    project_name: str,
    run_name: str,
    split_cache: Dict[Tuple, DataFrame] = None,
) -> ReportConditionData:
    """Get the results for a condition."""
    split_data: List[SplitConditionData] = []

    for split in splits:
        split = Split[split]
        if split == Split.inference:
            for inf_name in inference_names:
                split_data = _add_split_data_for_condition(
                    condition,
                    project_name,
                    run_name,
                    split,
                    split_data,
                    split_cache,
                    inference_name=inf_name,
                )
        else:
            split_data = _add_split_data_for_condition(
                condition, project_name, run_name, split, split_data, split_cache
            )

    return ReportConditionData(
        condition=_get_metric(condition),
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
        project_name=project_name,
        run_name=run_name,
        created_at=_get_email_datetime(),
        link=link,
        conditions=[],
    )

    # Build a cache of dataframes to avoid making multiple requests to the server
    split_cache: Dict[Tuple, DataFrame] = {}
    for c in conditions:
        report_data.conditions.append(
            _get_report_results_for_condition(
                c, logged_splits, inference_names, project_name, run_name, split_cache
            )
        )

    api_client.notify_email(report_data.dict(), "run_report", emails)
