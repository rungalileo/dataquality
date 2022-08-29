from typing import List

from vaex.dataframe import DataFrame

from dataquality.clients.api import ApiClient
from dataquality.clients.objectstore import ObjectStore
from dataquality.schemas.predicate import Predicate
from dataquality.utils.predicate_manager import PredicateManager

api_client = ApiClient()
object_store = ObjectStore()


def evaluate_predicate(df: DataFrame, predicate: Predicate) -> bool:
    """Evaluates a predicate on a given dataframe

    :param df: The dataframe
    :param predicate: Predicate to evaluate
    """
    pm = PredicateManager(df)
    return pm.evaluate_predicate(predicate)


def evaluate_predicates(df: DataFrame, predicates: List[Predicate]) -> List[bool]:
    """Evaluates a list of predicates on a given datafrme

    :param df: The dataframe
    :param predicates: List of predicates to evaluate
    """
    resp = []
    for predicate in predicates:
        resp.append(evaluate_predicate(df, predicate))

    return resp
