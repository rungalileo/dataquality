from typing import List
from uuid import UUID

from dataquality.clients.api import ApiClient
from dataquality.core.log import get_data_logger
from dataquality.metrics import get_dataframe
from dataquality.schemas.predicate import Predicate
from dataquality.schemas.split import Split

api_client = ApiClient()


def register_predicates(predicates: List[Predicate], emails: List[str]) -> None:
    get_data_logger().logger_config.predicates = predicates
    get_data_logger().logger_config.predicate_emails = emails


def evaluate_predicates(
    predicates: List[Predicate], emails: List[str], project_id: UUID, run_id: UUID
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
            for pred in predicates:
                pred_results = {}
                passes, val = pred.evaluate(df)
                pred_results.update(
                    {
                        "project_name": project_name,
                        "run_name": run_name,
                        "split": split,
                        "inference_name": inf_name,
                        "metric": pred.metric,
                        "agg": pred.agg.value,
                        "operator": pred.operator.value,
                        "threshold": pred.threshold,
                        "passes": passes,
                        "val": float(val),
                    }
                )
                results.append(pred_results)

        inference_names = [None]

    api_client.notify_predicates(results, emails)
