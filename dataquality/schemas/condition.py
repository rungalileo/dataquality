from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

from pydantic import BaseModel
from pydantic.class_validators import validator
from vaex.dataframe import DataFrame


class Operator(str, Enum):
    eq = "is equal to"
    neq = "is not equal to"
    gt = "is greater than"
    lt = "is less than"
    gte = "is greater than or equal to"
    lte = "is less than or equal to"


class AggregateFunction(str, Enum):
    avg = "Average"
    min = "Minimum"
    max = "Maximum"
    sum = "Sum"
    pct = "Percentage"


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


class ConditionFilter(BaseModel):
    """Filter a dataframe based on the column value

    Note that the column used for filtering is the same as the metric used
      in the condition.

    :param operator: The operator to use for filtering (e.g. "gt", "lt", "eq", "neq")
        See `Operator`
    :param value: The value to compare against
    """

    metric: str
    operator: Operator
    value: Union[float, int, str, bool]


class Condition(BaseModel):
    """Class for building custom conditions for data quality checks

    After building a condition, call `evaluate` to determine the truthiness
    of the condition against a given DataFrame.

    With a bit of thought, complex and custom conditions can be built. To gain an
    intuition for what can be accomplished, consider the following examples:

    1. Is the average confidence less than 0.3?
        >>> c = Condition(
        ...     agg=AggregateFunction.avg,
        ...     metric="confidence",
        ...     operator=Operator.lt,
        ...     threshold=0.3,
        ... )
        >>> c.evaluate(df)

    2. Is the max DEP greater or equal to 0.45?
        >>> c = Condition(
        ...     agg=AggregateFunction.max,
        ...     metric="data_error_potential",
        ...     operator=Operator.gte,
        ...     threshold=0.45,
        ... )
        >>> c.evaluate(df)

    By adding filters, you can further narrow down the scope of the condition.
    If the aggregate function is "pct", you don't need to specify a metric,
      as the filters will determine the percentage of data.
    For example:

    3. Alert if over 80% of the dataset has confidence under 0.1
        >>> c = Condition(
        ...     operator=Operator.gt,
        ...     threshold=0.8,
        ...     agg=AggregateFunction.pct,
        ...     filters=[
        ...         ConditionFilter(
        ...             metric="confidence", operator=Operator.lt, value=0.1
        ...         ),
        ...     ],
        ... )
        >>> c.evaluate(df)

    4. Alert if at least 20% of the dataset has drifted (Inference DataFrames only)
        >>> c = Condition(
        ...     operator=Operator.gte,
        ...     threshold=0.2,
        ...     agg=AggregateFunction.pct,
        ...     filters=[
        ...         ConditionFilter(
        ...             metric="is_drifted", operator=Operator.eq, value=True
        ...         ),
        ...     ],
        ... )
        >>> c.evaluate(df)

    5. Alert 5% or more of the dataset contains PII
        >>> c = Condition(
        ...     operator=Operator.gte,
        ...     threshold=0.05,
        ...     agg=AggregateFunction.pct,
        ...     filters=[
        ...         ConditionFilter(
        ...             metric="galileo_pii", operator=Operator.neq, value="None"
        ...         ),
        ...     ],
        ... )
        >>> c.evaluate(df)

    Complex conditions can be built when the filter has a different metric
    than the metric used in the condition. For example:

    6. Alert if the min confidence of drifted data is less than 0.15
        >>> c = Condition(
        ...     agg=AggregateFunction.min,
        ...     metric="confidence",
        ...     operator=Operator.lt,
        ...     threshold=0.15,
        ...     filters=[
        ...         ConditionFilter(
        ...             metric="is_drifted", operator=Operator.eq, value=True
        ...         )
        ...     ],
        ... )
        >>> c.evaluate(df)

    7. Alert if over 50% of high DEP (>=0.7) data contains PII
        >>> c = Condition(
        ...     operator=Operator.gt,
        ...     threshold=0.5,
        ...     agg=AggregateFunction.pct,
        ...     filters=[
        ...         ConditionFilter(
        ...             metric="data_error_potential", operator=Operator.gte, value=0.7
        ...         ),
        ...         ConditionFilter(
        ...             metric="galileo_pii", operator=Operator.neq, value="None"
        ...         ),
        ...     ],
        ... )
        >>> c.evaluate(df)

    You can also call conditions directly, which will assert its truth against a df
    1. Assert that average confidence less than 0.3
    >>> c = Condition(
    ...     agg=AggregateFunction.avg,
    ...     metric="confidence",
    ...     operator=Operator.lt,
    ...     threshold=0.3,
    ... )
    >>> c(df)  # Will raise an AssertionError if False


    :param metric: The DF column for evaluating the condition
    :param agg: An aggregate function to apply to the metric
    :param operator: The operator to use for comparing the agg to the threshold
        (e.g. "gt", "lt", "eq", "neq")
    :param threshold: Threshold value for evaluating the condition
    :param filter: Optional filter to apply to the DataFrame before evaluating the
        condition
    """

    agg: AggregateFunction
    operator: Operator
    threshold: float
    metric: Optional[str] = None
    filters: Optional[List[ConditionFilter]] = []

    def evaluate(self, df: DataFrame) -> Tuple[bool, float]:
        filtered_df = self._apply_filters(df)

        if self.agg == AggregateFunction.pct:
            value = AGGREGATE_FUNCTIONS[self.agg](filtered_df, df)
        else:
            value = AGGREGATE_FUNCTIONS[self.agg](filtered_df, self.metric)

        passes = CRITERIA_OPERATORS[self.operator](value, self.threshold)
        return passes, float(value)

    def _apply_filters(self, df: DataFrame) -> DataFrame:
        filtered_df = df.copy()

        filters = self.filters or []
        for filter in filters:
            filtered_df = FILTER_OPERATORS[filter.operator](
                filtered_df, filter.metric, filter.value
            )

        return filtered_df

    def __call__(self, df: DataFrame) -> None:
        """Asserts the condition"""
        assert self.evaluate(df)[0]

    @validator("filters", pre=True, always=True)
    def validate_filters(
        cls, v: Optional[List[ConditionFilter]], values: Dict
    ) -> Optional[List[ConditionFilter]]:
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
