from enum import Enum
from typing import Dict, List, Optional, Union

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

    :param operator: The operator to use for filtering (e.g. "gt", "lt", "eq", "neq")
        See `Operator`
    :param value: The value to compare against
    """

    metric: str
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
        ...     agg=AggregateFunction.avg,
        ...     metric="confidence",
        ...     operator=Operator.lt,
        ...     threshold=0.3,
        ... )
        >>> p.evaluate(df)

    2. Is the max DEP greater or equal to 0.45?
        >>> p = Predicate(
        ...     agg=AggregateFunction.max,
        ...     metric="data_error_potential",
        ...     operator=Operator.gte,
        ...     threshold=0.45,
        ... )
        >>> p.evaluate(df)

    By adding filters, you can further narrow down the scope of the predicate.
    If the aggregate function is "pct", you don't need to specify a metric,
      as the filters will determine the percentage of data.
    For example:

    3. Alert if over 80% of the dataset has confidence under 0.1
        >>> p = Predicate(
        ...     operator=Operator.gt,
        ...     threshold=0.8,
        ...     agg=AggregateFunction.pct,
        ...     filters=[
        ...         PredicateFilter(
        ...             metric="confidence", operator=Operator.lt, value=0.1
        ...         ),
        ...     ],
        ... )
        >>> p.evaluate(df)

    4. Alert if at least 20% of the dataset has drifted (Inference DataFrames only)
        >>> p = Predicate(
        ...     operator=Operator.gte,
        ...     threshold=0.2,
        ...     agg=AggregateFunction.pct,
        ...     filters=[
        ...         PredicateFilter(
        ...             metric="is_drifted", operator=Operator.eq, value=True
        ...         ),
        ...     ],
        ... )
        >>> p.evaluate(df)

    5. Alert 5% or more of the dataset contains PII
        >>> p = Predicate(
        ...     operator=Operator.gte,
        ...     threshold=0.05,
        ...     agg=AggregateFunction.pct,
        ...     filters=[
        ...         PredicateFilter(
        ...             metric="galileo_pii", operator=Operator.neq, value="None"
        ...         ),
        ...     ],
        ... )
        >>> p.evaluate(df)

    Complex predicates can be built when the filter has a different metric
    than the metric used in the predicate. For example:

    6. Alert if the min confidence of drifted data is less than 0.15
        >>> p = Predicate(
        ...     agg=AggregateFunction.min,
        ...     metric="confidence",
        ...     operator=Operator.lt,
        ...     threshold=0.15,
        ...     filters=[
        ...         PredicateFilter(
        ...             metric="is_drifted", operator=Operator.eq, value=True
        ...         )
        ...     ],
        ... )
        >>> p.evaluate(df)

    7. Alert if over 50% of high DEP (>=0.7) data contains PII
        >>> p = Predicate(
        ...     operator=Operator.gt,
        ...     threshold=0.5,
        ...     agg=AggregateFunction.pct,
        ...     filters=[
        ...         PredicateFilter(
        ...             metric="data_error_potential", operator=Operator.gte, value=0.7
        ...         ),
        ...         PredicateFilter(
        ...             metric="galileo_pii", operator=Operator.neq, value="None"
        ...         ),
        ...     ],
        ... )
        >>> p.evaluate(df)

    You can also call predicates directly, which will assert its truth against a df
    1. Assert that average confidence less than 0.3
    >>> p = Predicate(
    ...     agg=AggregateFunction.avg,
    ...     metric="confidence",
    ...     operator=Operator.lt,
    ...     threshold=0.3,
    ... )
    >>> p(df)  # Will raise an AssertionError if False


    :param metric: The DF column for evaluating the predicate
    :param agg: An aggregate function to apply to the metric
    :param operator: The operator to use for comparing the agg to the threshold
        (e.g. "gt", "lt", "eq", "neq")
    :param threshold: Threshold value for evaluating the predicate
    :param filter: Optional filter to apply to the DataFrame before evaluating the
        predicate
    """

    agg: AggregateFunction
    operator: Operator
    threshold: float
    metric: Optional[str] = None
    filters: Optional[List[PredicateFilter]] = []

    def evaluate(self, df: DataFrame) -> bool:
        filtered_df = self._apply_filters(df)

        if self.agg == AggregateFunction.pct:
            value = AGGREGATE_FUNCTIONS[self.agg](filtered_df, df)
        else:
            value = AGGREGATE_FUNCTIONS[self.agg](filtered_df, self.metric)

        return CRITERIA_OPERATORS[self.operator](value, self.threshold)

    def _apply_filters(self, df: DataFrame) -> DataFrame:
        filtered_df = df.copy()

        filters = self.filters or []
        for filter in filters:
            filtered_df = FILTER_OPERATORS[filter.operator](
                filtered_df, filter.metric, filter.value
            )

        return filtered_df

    def __call__(self, df: DataFrame) -> None:
        """Asserts the predicate"""
        assert self.evaluate(df)

    @validator("filters", pre=True, always=True)
    def validate_filters(
        cls, v: Optional[List[PredicateFilter]], values: Dict
    ) -> Optional[List[PredicateFilter]]:
        if not v:
            agg = values["agg"]
            if agg == AggregateFunction.pct:
                raise ValueError("Percentage aggregate requires a filter")

        return v

    @validator("metric", pre=True, always=True)
    def validate_metric(cls, v: Optional[str], values: Dict) -> Optional[str]:
        if not v:
            agg = values["agg"]
            if agg != AggregateFunction.pct:
                raise ValueError(
                    f"You must set a metric for non-percentage aggregate function {agg}"
                )

        return v
