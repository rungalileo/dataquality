from enum import Enum
from typing import Optional, Union

from pydantic import BaseModel


class Operator(str, Enum):
    eq = "eq"
    neq = "neq"
    gt = "gt"
    lt = "lt"
    gte = "gte"
    lte = "lte"


OPERATORS = {
    Operator.eq: lambda df, col, val: df[df[col] == val],
    Operator.neq: lambda df, col, val: df[df[col] != val],
    Operator.gt: lambda df, col, val: df[df[col] > val],
    Operator.lt: lambda df, col, val: df[df[col] < val],
    Operator.gte: lambda df, col, val: df[df[col] >= val],
    Operator.lte: lambda df, col, val: df[df[col] <= val],
}


class Critereon(BaseModel):
    type: str
    value: Union[str, int, float]
    threshold: Optional[float] = None


class Predicate(BaseModel):
    metric: str
    operator: Operator
    threshold: float
    critereon: Critereon
