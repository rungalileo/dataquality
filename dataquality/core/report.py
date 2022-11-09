from typing import List
from uuid import UUID

from dataquality.clients.api import ApiClient
from dataquality.core.log import get_data_logger
from dataquality.metrics import get_dataframe
from dataquality.schemas.condition import Condition
from dataquality.schemas.split import Split

api_client = ApiClient()


def build_run_report(conditions: List[Condition], emails: List[str]) -> None:
    get_data_logger().logger_config.conditions = conditions
    get_data_logger().logger_config.report_emails = emails


def evaluate_predicates(
    conditions: List[Condition], emails: List[str], project_id: UUID, run_id: UUID
) -> None:
    project_name = api_client.get_project(project_id)["name"]
    run_name = api_client.get_project_run(project_id, run_id)["name"]
    logged_splits = api_client.get_splits(project_id, run_id)

    results = []
    inference_names = [None]
    for split in logged_splits["splits"]:
        if split == Split.inference:
            inference_names = api_client.get_inference_names(project_id, run_id)

        for inf_name in inference_names:
            df = get_dataframe(project_name, run_name, split, inf_name, as_pandas=False)
            for c in conditions:
                results = {}
                passes, val = c.evaluate(df)
                results.update(
                    {
                        "project_name": project_name,
                        "run_name": run_name,
                        "split": split,
                        "inference_name": inf_name,
                        "metric": c.metric,
                        "agg": c.agg.value,
                        "operator": c.operator.value,
                        "threshold": c.threshold,
                        "passes": passes,
                        "val": float(val),
                    }
                )
                results.append(results)

        inference_names = [None]

    api_client.notify_email(results, "run_report", emails)
