from typing import Optional

from vaex.dataframe import DataFrame

from dataquality.schemas.predicate import AggregateFunction, Operator, Predicate

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


class PredicateManager:
    def apply_filter(self, df: DataFrame, predicate: Predicate) -> Optional[DataFrame]:
        df = df.copy()

        filter = predicate.filter
        if filter:
            return FILTER_OPERATORS[filter.operator](df, predicate.metric, filter.value)

        return df

    def evaluate_predicate(self, df: DataFrame, predicate: Predicate) -> bool:
        filter_df = self.apply_filter(df, predicate)
        if predicate.agg == AggregateFunction.pct:
            value = AGGREGATE_FUNCTIONS[predicate.agg](filter_df, df)
        else:
            value = AGGREGATE_FUNCTIONS[predicate.agg](filter_df, predicate.metric)

        return CRITERIA_OPERATORS[predicate.operator](value, predicate.threshold)
