---
description: A class to build custom conditions for DataFrame assertions and alerting.
---

# Build your own Conditions

A `Condition` is a class for building custom data quality checks.  Simply create a condition, and after the run is processed your conditions will be evaluated. Integrate with email or slack to have condition results alerting via a Run Report. Use Conditions to answer questions such as "Is the average confidence for my training data below 0.25" or "Has over 20% of my inference data drifted".



### What do I do with Conditions?

You can build a `Run Report` that will evaluate all conditions after a run is processed.

```python
import dataquality as dq

dq.init("text_classification")

cond1 = dq.Condition(...)
cond2 = dq.Condition(...)
dq.register_run_report(conditions=[cond1, cond2])

# By default we email the logged in user
# Optionally pass in additional emails to receive Run Reports
dq.register_run_report(conditions=[cond1], emails=["foo@bar.com"]
```

You can also build and evaluate conditions by accessing the processed DataFrame.&#x20;

```python
from dataquality import Condition

df = dq.metrics.get_dataframe("proj_name", "run_name", "training")
cond = Condition(...)
passes, ground_truth = cond.evaluate(df)
```



### How do I build a Condition?

A `Condition` is defined as follows:

\`\`\`

```python
class Condition:
    agg: AggregateFunction # An aggregate function to apply to the metric
    threshold: float # Threshold value for evaluating the condition
    operator: Operator # The operator to use for comparing the agg to the threshold
    metric: Optional[str] = None # The DF column for evaluating the condition
    filters: Optional[List[ConditionFilter]] = [] # Optional filter to apply to the DataFrame before evaluating the Condition
```

To gain an intuition for what can be accomplished, consider the following examples:

```
1. Is the average confidence less than 0.3?
    >>> c = Condition(
    ...     agg=AggregateFunction.avg,
    ...     metric="confidence",
    ...     operator=Operator.lt,
    ...     threshold=0.3,
    ... )

2. Is the max DEP greater or equal to 0.45?
    >>> c = Condition(
    ...     agg=AggregateFunction.max,
    ...     metric="data_error_potential",
    ...     operator=Operator.gte,
    ...     threshold=0.45,
    ... )
```

By adding filters, you can further narrow down the scope of the condition. If the aggregate function is "pct", you don't need to specify a metric, as the filters will determine the percentage of data.&#x20;

```
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
```

Complex conditions can be built when the filter has a different metric than the metric used in the condition.

```
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
```

You can also call conditions directly, which will assert its truth against a DataFrame.

```
1. Assert that average confidence less than 0.3
>>> c = Condition(
...     agg=AggregateFunction.avg,
...     metric="confidence",
...     operator=Operator.lt,
...     threshold=0.3,
... )
>>> c(df)  # Will raise an AssertionError if False
```

### Aggregate Function

```python
from dataquality import AggregateFunction
```

The available aggregate functions are:

```python
class AggregateFunction(str, Enum):
    avg = "avg"
    min = "min"
    max = "max"
    sum = "sum"
    pct = "pct"
```

### Operator

```python
from dataquality import Operator
```

The available operators are:

```python
class Operator(str, Enum):
    eq = "eq"
    neq = "neq"
    gt = "gt"
    lt = "lt"
    gte = "gte"
    lte = "lte"
```

### Metric & Treshold

The metric must be the name of a column in the DataFrame. Threshold is a numeric value for comparison in the Condition.&#x20;

### Alerting

Alerting via email, slack in development. Please reach out to Galileo at team@rungalileo.io for more information.
