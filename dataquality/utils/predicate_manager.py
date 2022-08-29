from vaex.dataframe import DataFrame

from dataquality.schemas.predicate import Predicate


class PredicateManager:
    def __init__(self, df: DataFrame) -> None:
        pass

    def evaluate_predicate(self, predicate: Predicate) -> bool:
        return False
