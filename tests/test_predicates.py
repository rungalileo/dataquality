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
    """Average confidence compared to 0.3"""
    inp = dict(
        id=range(0, 10),
        confidence=[0.1] * 10,
    )  # True avg confidence is 0.1
    df = vaex.from_dict(inp)
    p = Predicate(
        agg=AggregateFunction.avg,
        metric="confidence",
        operator=operator,
        threshold=0.3,
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
    """Max DEP compared to 0.45"""
    inp = dict(
        id=range(0, 10),
        dep=[0.3] * 9 + [0.45],
    )  # True max DEP is 0.45
    df = vaex.from_dict(inp)
    p = Predicate(
        agg=AggregateFunction.max,
        metric="dep",
        operator=operator,
        threshold=0.45,
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
    """80% of dataset has confidence less than 0.1"""
    inp = dict(
        id=range(0, 10),
        confidence=[0.05] * 9 + [0.8],
    )  # True value is 90%
    df = vaex.from_dict(inp)
    p = Predicate(
        operator=operator,
        threshold=0.8,
        agg=AggregateFunction.pct,
        metric="confidence",
        filters=[PredicateFilter(metric="confidence", operator=Operator.lt, value=0.1)],
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
        (Operator.lt, False),
        (Operator.lte, False),
        (Operator.gte, True),
        (Operator.gt, True),
    ],
)
def test_evaluate_predicate_5(operator: Operator, expected: bool) -> None:
    """Pct of dataset that contains PII"""
    inp = dict(
        id=range(0, 10),
        galileo_pii=["PII"] * 2 + ["None"] * 8,
    )  # True pct if 20%
    df = vaex.from_dict(inp)
    p = Predicate(
        operator=operator,
        threshold=0.05,
        agg=AggregateFunction.pct,
        filters=[
            PredicateFilter(metric="galileo_pii", operator=Operator.neq, value="None")
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
def test_evaluate_predicate_6(operator: Operator, expected: bool) -> None:
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
def test_evaluate_predicate_7(operator: Operator, expected: bool) -> None:
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


def test_predicate_missing_metric_for_pct_agg() -> None:
    with pytest.raises(ValueError):
        Predicate(
            agg=AggregateFunction.avg,
            operator=Operator.gt,
            threshold=0.2,
        )


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
def test_evaluate_predicate_call(operator: Operator, expected: bool) -> None:
    """Make sure that on failures, the predicate raises an AssertionError"""
    inp = dict(
        id=range(0, 10),
        confidence=[0.1] * 10,
    )
    df = vaex.from_dict(inp)
    p = Predicate(
        agg=AggregateFunction.avg,
        metric="confidence",
        operator=operator,
        threshold=0.3,
    )
    if expected is True:
        p(df)
    else:
        with pytest.raises(AssertionError):
            p(df)
