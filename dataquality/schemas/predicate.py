from enum import Enum
from typing import Optional, Union

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
    Operator.eq: lambda val, threshold: val == threshold,
    Operator.neq: lambda val, threshold: val != threshold,
    Operator.gt: lambda val, threshold: val > threshold,
    Operator.lt: lambda val, threshold: val < threshold,
    Operator.gte: lambda val, threshold: val >= threshold,
    Operator.lte: lambda val, threshold: val <= threshold,
}



class AggregateFunction(str, Enum):
    avg = "avg"
    min = "min"
    max = "max"
    sum = "sum"
    pct = "pct"


AGGREGATE_FUNCTIONS = {
    AggregateFunction.avg: lambda df, col: df.mean(col),
    AggregateFunction.min: lambda df, col: df.min(col),
    AggregateFunction.max: lambda df, col: df.max(col),
    AggregateFunction.sum: lambda df, col: df.sum(col),
    AggregateFunction.pct: lambda df, df2: df.count() / df2.count(),
}


class PredicateFilter(BaseModel):
    """A class for representing a metric to be evaluated on a dataframe

    Args:
        col: The name of the DF column to evaluate on
        operator: The operator to use for the evaluation
        value: The value to use for the evaluation

    E.g. 
    """
    operator: Operator
    value: Union[int, float, str]

    def evaluate(self, df: DataFrame, col: str) -> DataFrame:
        return FILTER_OPERATORS[self.operator](df, col, self.value)


class Predicate(BaseModel):
    filter: Optional[PredicateFilter] = None
    col: str  # metric
    agg: AggregateFunction
    criteria_operator: Operator
    criteria_threshold: float


