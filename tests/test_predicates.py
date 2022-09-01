import pytest
import vaex

from dataquality import AggregateFunction, Operator, Predicate, PredicateFilter


@pytest.mark.parametrize(
    "operator,expected",
    [
        (Operator.eq, False),
        (Operator.neq, True),
        (Operator.lt, True),
        (Operator.lte, True),
        (Operator.gte, False),
        (Operator.gt, False),
    ],
)
def test_evaluate_predicate_1(operator: Operator, expected: bool) -> None:
    """Average confidence compared to 0.8"""
    inp = dict(
        id=range(0, 10),
        confidence=[0.1] * 10,
    )  # True avg confidence is 0.1
    df = vaex.from_dict(inp)
    p = Predicate(
        agg=AggregateFunction.avg,
        metric="confidence",
        operator=operator,
        threshold=0.8,
    )
    assert p.evaluate(df) is expected


@pytest.mark.parametrize(
    "operator,expected",
    [
        (Operator.eq, True),
        (Operator.neq, False),
        (Operator.lt, False),
        (Operator.lte, True),
        (Operator.gte, True),
        (Operator.gt, False),
    ],
)
def test_evaluate_predicate_2(operator: Operator, expected: bool) -> None:
    """Max DEP compared to 0.35"""
    inp = dict(
        id=range(0, 10),
        dep=[0.3] * 9 + [0.35],
    )  # True max DEP is 0.35
    df = vaex.from_dict(inp)
    p = Predicate(
        operator=operator,
        agg=AggregateFunction.max,
        metric="dep",
        threshold=0.35,
    )
    assert p.evaluate(df) is expected


@pytest.mark.parametrize(
    "operator,expected",
    [
        (Operator.eq, False),
        (Operator.neq, True),
        (Operator.lt, False),
        (Operator.lte, False),
        (Operator.gte, True),
        (Operator.gt, True),
    ],
)
def test_evaluate_predicate_3(operator: Operator, expected: bool) -> None:
    """60% of dataset has confidence less than 0.3"""
    inp = dict(
        id=range(0, 10),
        confidence=[0.2] * 7 + [0.8] * 3,
    )  # True value is 70%
    df = vaex.from_dict(inp)
    p = Predicate(
        operator=operator,
        threshold=0.6,
        agg=AggregateFunction.pct,
        metric="confidence",
        filters=[PredicateFilter(metric="confidence", operator=Operator.lt, value=0.3)],
    )
    assert p.evaluate(df) is expected


@pytest.mark.parametrize(
    "operator,expected",
    [
        (Operator.eq, True),
        (Operator.neq, False),
        (Operator.lt, False),
        (Operator.lte, True),
        (Operator.gte, True),
        (Operator.gt, False),
    ],
)
def test_evaluate_predicate_4(operator: Operator, expected: bool) -> None:
    """20% of the inference dataset has drifted"""
    inp = dict(
        id=range(0, 10),
        is_drifted=[True] * 2 + [False] * 8,
    )  # True value is 20%
    df = vaex.from_dict(inp)
    p = Predicate(
        operator=operator,
        threshold=0.2,
        agg=AggregateFunction.pct,
        metric="is_drifted",
        filters=[
            PredicateFilter(metric="is_drifted", operator=Operator.eq, value=True)
        ],
    )
    assert p.evaluate(df) is expected


@pytest.mark.parametrize(
    "operator,expected",
    [
        (Operator.eq, False),
        (Operator.neq, True),
        (Operator.lt, True),
        (Operator.lte, True),
        (Operator.gte, False),
        (Operator.gt, False),
    ],
)
def test_evaluate_predicate_5(operator: Operator, expected: bool) -> None:
    """Min confidence of drifted data compared to 0.15"""
    inp = dict(
        id=range(0, 10),
        is_drifted=[True] * 2 + [False] * 8,
        confidence=[0.1, 0.2] + [0.6] * 8,
    )  # True min confidence is 0.1
    df = vaex.from_dict(inp)
    p = Predicate(
        operator=operator,
        threshold=0.15,
        agg=AggregateFunction.min,
        metric="confidence",
        filters=[
            PredicateFilter(metric="is_drifted", operator=Operator.eq, value=True)
        ],
    )
    assert p.evaluate(df) is expected


@pytest.mark.parametrize(
    "operator,expected",
    [
        (Operator.eq, True),
        (Operator.neq, False),
        (Operator.lt, False),
        (Operator.lte, True),
        (Operator.gte, True),
        (Operator.gt, False),
    ],
)
def test_evaluate_predicate_6(operator: Operator, expected: bool) -> None:
    """Pct of high DEP (>=0.7) dataset that contains PII"""
    inp = dict(
        id=range(0, 10),
        galileo_pii=["name"] * 4 + ["None"] * 6,
        data_error_potential=[0.8, 0.7, 0.6, 0.5] + [0.9] * 6,
    )  # True value is 0.2
    df = vaex.from_dict(inp)
    p = Predicate(
        agg=AggregateFunction.pct,
        operator=operator,
        threshold=0.2,
        filters=[
            PredicateFilter(
                metric="data_error_potential", operator=Operator.gte, value=0.7
            ),
            PredicateFilter(metric="galileo_pii", operator=Operator.neq, value="None"),
        ],
    )
    assert p.evaluate(df) is expected


def test_predicate_pct_agg_requires_filter() -> None:
    with pytest.raises(ValueError):
        Predicate(
            metric="is_drifted",
            agg=AggregateFunction.pct,
            operator=Operator.gt,
            threshold=0.2,
        )
