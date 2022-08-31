from enum import Enum
from typing import Dict, Optional, Union

from pydantic import BaseModel
from pydantic.class_validators import validator


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


class Predicate(BaseModel):
    metric: str
    agg: AggregateFunction
    operator: Operator
    threshold: float
    filter: Optional[PredicateFilter] = None

    @validator("filter", pre=True, always=True)
    def validate_filter(
        cls, v: Optional[PredicateFilter], values: Dict
    ) -> Optional[PredicateFilter]:
        if not v:
            agg = values["agg"]
            if agg == AggregateFunction.pct:
                raise ValueError("Percentage aggregate requires a filter")

        return v
