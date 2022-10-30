from typing import List
from vaex.dataframe import DataFrame

from dataquality.core.log import get_data_logger
from dataquality.schemas.predicate import Predicate


def register_predicates(predicates: List[Predicate]) -> None:
    get_data_logger().logger_config.predicates = predicates


def evaluate_predicates(
    predicates: List[Predicate], project_name: Optional[str] = None, run_name: Optional[str] = None
) -> None:
    for pred in predicates:
        pred.evaluate(df)
