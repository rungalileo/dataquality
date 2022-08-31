from enum import Enum
from typing import Dict, Optional, Union

from pydantic import BaseModel
from pydantic.class_validators import validator
from vaex.dataframe import DataFrame


class Operator(str, Enum):
    eq = "eq"
    neq = "neq"
    gt = "gt"
    lt = "lt"
    gte = "gte"
    lte = "lte"


class AggregateFunction(str, Enum):
    avg = "avg"
    min = "min"
    max = "max"
    sum = "sum"
    pct = "pct"


class PredicateFilter(BaseModel):
    """A class for representing a metric to be evaluated on a dataframe

    Args:
        col: The name of the DF column to evaluate on
        operator: The operator to use for the evaluation
        value: The value to use for the evaluation

    E.g.
    """

    operator: Operator
    value: Union[float, int, str, bool]


# Filter a dataframe based on a column value
FILTER_OPERATORS = {
    Operator.eq: lambda df, col, val: df[df[col] == val],
    Operator.neq: lambda df, col, val: df[df[col] != val],
    Operator.gt: lambda df, col, val: df[df[col] > val],
    Operator.lt: lambda df, col, val: df[df[col] < val],
    Operator.gte: lambda df, col, val: df[df[col] >= val],
    Operator.lte: lambda df, col, val: df[df[col] <= val],
}


# Returns boolean of a value compared to a threshold
CRITERIA_OPERATORS = {
    Operator.eq: lambda val, threshold: bool(val == threshold),
    Operator.neq: lambda val, threshold: bool(val != threshold),
    Operator.gt: lambda val, threshold: bool(val > threshold),
    Operator.lt: lambda val, threshold: bool(val < threshold),
    Operator.gte: lambda val, threshold: bool(val >= threshold),
    Operator.lte: lambda val, threshold: bool(val <= threshold),
}


AGGREGATE_FUNCTIONS = {
    AggregateFunction.avg: lambda df, col: df.mean(col),
    AggregateFunction.min: lambda df, col: df.min(col),
    AggregateFunction.max: lambda df, col: df.max(col),
    AggregateFunction.sum: lambda df, col: df.sum(col),
    AggregateFunction.pct: lambda df, df2: df.count() / df2.count(),
}


class Predicate(BaseModel):
    metric: str
    agg: AggregateFunction
    operator: Operator
    threshold: float
    filter: Optional[PredicateFilter] = None

    def evaluate(self, df: DataFrame) -> bool:
        filter_df = self.apply_filter(df)
        if self.agg == AggregateFunction.pct:
            value = AGGREGATE_FUNCTIONS[self.agg](filter_df, df)
        else:
            value = AGGREGATE_FUNCTIONS[self.agg](filter_df, self.metric)

        return CRITERIA_OPERATORS[self.operator](value, self.threshold)

    def apply_filter(self, df: DataFrame) -> Optional[DataFrame]:
        df = df.copy()

        filter = self.filter
        if filter:
            return FILTER_OPERATORS[filter.operator](df, self.metric, filter.value)

        return df

    @validator("filter", pre=True, always=True)
    def validate_filter(
        cls, v: Optional[PredicateFilter], values: Dict
    ) -> Optional[PredicateFilter]:
        if not v:
            agg = values["agg"]
            if agg == AggregateFunction.pct:
                raise ValueError("Percentage aggregate requires a filter")

        return v
