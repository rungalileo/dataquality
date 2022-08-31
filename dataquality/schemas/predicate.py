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


class PredicateFilter(BaseModel):
    """Filter a dataframe based on the column value

    Note that the column used for filtering is the same as the metric used
      in the predicate.

    :param operator: The operator to use for filtering (e.g. >, <, ==, !=)
    :param value: The value to compare against
    """

    operator: Operator
    value: Union[float, int, str, bool]


class Predicate(BaseModel):
    """Class for building custom predicates for data quality checks

    After building a predicate, call `evaluate` to determine the truthiness
    of the predicate against a given DataFrame.

    With a bit of thought, complex and custom predicates can be built. To gain an
    intuition for what can be accomplished, consider the following examples:

    1. Is the average confidence less than 0.3?
        >>> p = Predicate(
        ...     metric="confidence",
        ...     agg=AggregateFunction.avg,
        ...     operator=Operator.lt,
        ...     threshold=0.3,
        ... )
        >>> p.evaluate(df)
        True

    2. Is the max DEP greater or equal to 0.45?
        >>> p = Predicate(
        ...     metric="confidence",
        ...     agg=AggregateFunction.max,
        ...     operator=Operator.gte,
        ...     threshold=0.45,
        ... )
        >>> p.evaluate(df)
        True

    By adding filters, you can further narrow down the scope of the predicate.
    For example:

    3. Alert if over 80% of the dataset has confidence under 0.1
        >>> p = Predicate(
        ...     filter=PredicateFilter(operator=Operator.lt, value=0.1),
        ...     metric="confidence",
        ...     agg=AggregateFunction.pct,
        ...     operator=Operator.gt,
        ...     threshold=0.8,
        ... )
        >>> p.evaluate(df)
        True

    4. Alert if at least 20% of the dataset has drifted (Inference DataFrames only)
        >>> p = Predicate(
        ...     filter=PredicateFilter(operator=Operator.eq, value=True),
        ...     metric="is_drifted",
        ...     agg=AggregateFunction.pct,
        ...     operator=Operator.gte,
        ...     threshold=0.20,
        ... )
        >>> p.evaluate(df)
        True

    5. Alert 5% or more of the dataset contains PII
        >>> p = Predicate(
        ...     filter=PredicateFilter(operator=Operator.neq, value=None),
        ...     metric="galileo_pii",
        ...     agg=AggregateFunction.pct,
        ...     operator=Operator.gte,
        ...     threshold=0.05,
        ... )
        >>> p.evaluate(df)
        True

    :param metric: The DF column for evaluating the predicate
    :param agg: An aggregate function to apply to the metric
    :param operator: The operator to use for comparing the agg to the threshold
        (e.g. >, <, ==, !=)
    :param threshold: Threshold value for evaluating the predicate
    :param filter: Optional filter to apply to the DataFrame before evaluating the
        predicate
    """

    metric: str
    agg: AggregateFunction
    operator: Operator
    threshold: float
    filter: Optional[PredicateFilter] = None

    def evaluate(self, df: DataFrame) -> bool:
        filter_df = self._apply_filter(df)
        if self.agg == AggregateFunction.pct:
            value = AGGREGATE_FUNCTIONS[self.agg](filter_df, df)
        else:
            value = AGGREGATE_FUNCTIONS[self.agg](filter_df, self.metric)

        return CRITERIA_OPERATORS[self.operator](value, self.threshold)

    def _apply_filter(self, df: DataFrame) -> Optional[DataFrame]:
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
