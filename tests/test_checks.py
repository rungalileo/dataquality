import pytest
import vaex

from dataquality.checks import evaluate_predicate
from dataquality.schemas.predicate import (
    AggregateFunction,
    Operator,
    Predicate,
    PredicateFilter,
)


def test_evaluate_predicate_1():
    """Average confidence is less than or equal to 0.8"""
    inp = dict(
        id=range(0, 10),
        confidence=[0.1] * 10,
    )
    df = vaex.from_dict(inp)
    pr = Predicate(
        filter=None,
        metric="confidence",
        agg=AggregateFunction.avg,
        operator=Operator.lte,
        threshold=0.8,
    )
    assert evaluate_predicate(df, pr) is True


def test_evaluate_predicate_2():
    """Max DEP is greater than 0.35"""
    inp = dict(
        id=range(0, 10),
        dep=[0.3] * 9 + [0.4],
    )
    df = vaex.from_dict(inp)
    pr = Predicate(
        filter=None,
        metric="dep",
        agg=AggregateFunction.max,
        operator=Operator.gt,
        threshold=0.35,
    )
    assert evaluate_predicate(df, pr) is True


def test_evaluate_predicate_3():
    """Over 60% of dataset has confidence less than 0.3"""
    inp = dict(
        id=range(0, 10),
        confidence=[0.2] * 7 + [0.8] * 3,
    )
    df = vaex.from_dict(inp)
    pr = Predicate(
        filter=PredicateFilter(operator=Operator.lt, value=0.3),
        metric="confidence",
        agg=AggregateFunction.pct,
        operator=Operator.gt,
        threshold=0.6,
    )
    assert evaluate_predicate(df, pr) is True


def test_evaluate_predicate_4():
    """At least 20% of the inference dataset has drifted"""
    inp = dict(
        id=range(0, 10),
        is_drifted=[True] * 5 + [False] * 5,
    )
    df = vaex.from_dict(inp)
    pr = Predicate(
        filter=PredicateFilter(operator=Operator.eq, value=True),
        metric="is_drifted",
        agg=AggregateFunction.pct,
        operator=Operator.gt,
        threshold=0.2,
    )
    assert evaluate_predicate(df, pr) is True


def test_predicate_pct_agg_requires_filter() -> None:
    with pytest.raises(ValueError):
        Predicate(
            metric="is_drifted",
            agg=AggregateFunction.pct,
            operator=Operator.gt,
            threshold=0.2,
        )
