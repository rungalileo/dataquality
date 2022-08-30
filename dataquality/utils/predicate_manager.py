from typing import Optional
from vaex.dataframe import DataFrame

from dataquality.schemas.predicate import FILTER_OPERATORS, OPERATORS, Criteria, Operator, Predicate, PredicateFilter


class PredicateManager:
    def apply_filter(self, df: DataFrame, predicate: Predicate) -> Optional[DataFrame]:
        filter = predicate.filter

        if filter:
            return FILTER_OPERATORS[filter.operator](df, predicate.col, filter.value)

        return df

    def evaluate_predicate(self, df: DataFrame, predicate: Predicate) -> bool:
        filter_df = self.apply_filter(df, predicate)
        return


"""
Alert when:
1. Average confidence is greater than or equal to 0.8
2. Max DEP is less than 0.35
3. greater than 0.6 of the rows have confidence less than 0.3
4. more than 20% of the inference dataset has drifted

pr1 = Predicate(
    filter=None,
    col="confidence",
    type=Operator.avg,
    criteria_operator=Operator.gte,
    criteria_threshold=0.8
)

pr2 = Predicate(
    filter=None,
    col="dep",
    type=Operator.max,
    criteria_operator=Operator.lt,
    criteria_threshold=0.35,
)

pr3 = Predicate(
    filter=PredicateFilter(
        operator=Operator.lt,
        value=0.3
    ),
    col="confidence",
    type=Operator.pct,
    criteria_operator=Operator.lt,
    criteria_threshold=0.6
)

pr4 = Predicate(
    filter=PredicateFilter(
        operator=Operator.eq,
        value=True
    ),
    col="is_drifted",
    type=Operator.pct,
    criteria_operator=Operator.gt,
    criteria_threshold=0.2
)

"""