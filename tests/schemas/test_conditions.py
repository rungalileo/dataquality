import numpy as np
import pytest
import vaex

from dataquality import AggregateFunction, Condition, ConditionFilter, Operator


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
def test_evaluate_condition_1(operator: Operator, expected: bool) -> None:
    """Average confidence compared to 0.3"""
    inp = dict(
        id=range(0, 10),
        confidence=[0.1] * 10,
    )  # True avg confidence is 0.1
    df = vaex.from_dict(inp)
    c = Condition(
        agg=AggregateFunction.avg,
        metric="confidence",
        operator=operator,
        threshold=0.3,
    )
    passes, val = c.evaluate(df)
    assert passes is expected
    assert np.isclose(val, 0.1)


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
def test_evaluate_condition_2(operator: Operator, expected: bool) -> None:
    """Max DEP compared to 0.45"""
    inp = dict(
        id=range(0, 10),
        dep=[0.3] * 9 + [0.45],
    )  # True max DEP is 0.45
    df = vaex.from_dict(inp)
    c = Condition(
        agg=AggregateFunction.max,
        metric="dep",
        operator=operator,
        threshold=0.45,
    )
    passes, val = c.evaluate(df)
    assert passes is expected
    assert val == 0.45


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
def test_evaluate_condition_3(operator: Operator, expected: bool) -> None:
    """80% of dataset has confidence less than 0.1"""
    inp = dict(
        id=range(0, 10),
        confidence=[0.05] * 9 + [0.8],
    )  # True value is 90%
    df = vaex.from_dict(inp)
    c = Condition(
        operator=operator,
        threshold=0.8,
        agg=AggregateFunction.pct,
        metric="confidence",
        filters=[ConditionFilter(metric="confidence", operator=Operator.lt, value=0.1)],
    )
    passes, val = c.evaluate(df)
    assert passes is expected
    assert val == 0.90


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
def test_evaluate_condition_4(operator: Operator, expected: bool) -> None:
    """20% of the inference dataset has drifted"""
    inp = dict(
        id=range(0, 10),
        is_drifted=[True] * 2 + [False] * 8,
    )  # True value is 20%
    df = vaex.from_dict(inp)
    c = Condition(
        operator=operator,
        threshold=0.2,
        agg=AggregateFunction.pct,
        filters=[
            ConditionFilter(metric="is_drifted", operator=Operator.eq, value=True)
        ],
    )
    passes, val = c.evaluate(df)
    assert passes is expected
    assert val == 0.20


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
def test_evaluate_condition_5(operator: Operator, expected: bool) -> None:
    """Pct of dataset that contains PII"""
    inp = dict(
        id=range(0, 10),
        galileo_pii=["PII"] * 2 + ["None"] * 8,
    )  # True pct if 20%
    df = vaex.from_dict(inp)
    c = Condition(
        operator=operator,
        threshold=0.05,
        agg=AggregateFunction.pct,
        filters=[
            ConditionFilter(metric="galileo_pii", operator=Operator.neq, value="None")
        ],
    )
    passes, val = c.evaluate(df)
    assert passes is expected
    assert val == 0.20


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
def test_evaluate_condition_6(operator: Operator, expected: bool) -> None:
    """Min confidence of drifted data compared to 0.15"""
    inp = dict(
        id=range(0, 10),
        is_drifted=[True] * 2 + [False] * 8,
        confidence=[0.1, 0.2] + [0.6] * 8,
    )  # True min confidence is 0.1
    df = vaex.from_dict(inp)
    c = Condition(
        operator=operator,
        threshold=0.15,
        agg=AggregateFunction.min,
        metric="confidence",
        filters=[
            ConditionFilter(metric="is_drifted", operator=Operator.eq, value=True)
        ],
    )
    passes, val = c.evaluate(df)
    assert passes is expected
    assert val == 0.1


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
def test_evaluate_condition_7(operator: Operator, expected: bool) -> None:
    """Pct of high DEP (>=0.7) dataset that contains PII"""
    inp = dict(
        id=range(0, 10),
        galileo_pii=["name"] * 4 + ["None"] * 6,
        data_error_potential=[0.8, 0.7, 0.6, 0.5] + [0.9] * 6,
    )  # True value is 0.2
    df = vaex.from_dict(inp)
    c = Condition(
        agg=AggregateFunction.pct,
        operator=operator,
        threshold=0.2,
        filters=[
            ConditionFilter(
                metric="data_error_potential", operator=Operator.gte, value=0.7
            ),
            ConditionFilter(metric="galileo_pii", operator=Operator.neq, value="None"),
        ],
    )
    passes, val = c.evaluate(df)
    assert passes is expected
    assert val == 0.20


def test_condition_pct_agg_requires_filter() -> None:
    with pytest.raises(ValueError):
        Condition(
            metric="is_drifted",
            agg=AggregateFunction.pct,
            operator=Operator.gt,
            threshold=0.2,
        )


def test_condition_missing_metric_for_pct_agg() -> None:
    with pytest.raises(ValueError):
        Condition(
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
def test_evaluate_condition_call(operator: Operator, expected: bool) -> None:
    """Make sure that on failures, the condition raises an AssertionError"""
    inp = dict(
        id=range(0, 10),
        confidence=[0.1] * 10,
    )
    df = vaex.from_dict(inp)
    c = Condition(
        agg=AggregateFunction.avg,
        metric="confidence",
        operator=operator,
        threshold=0.3,
    )
    if expected is True:
        c(df)
    else:
        with pytest.raises(AssertionError):
            c(df)
