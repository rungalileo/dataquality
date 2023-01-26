# 🧪 Automated NLP Production Monitoring Powered By Galileo Conditions

<figure><img src="../../../.gitbook/assets/Screen Shot 2022-09-15 at 11.12.03 PM.png" alt=""><figcaption></figcaption></figure>

Leverage all the Galileo 'building blocks' that are logged and stored for you to create Tests using Galileo Conditions -- a class for building custom data quality checks.

Conditions are simple and flexible, allowing you to author powerful data/model tests.&#x20;

### Run Report

Integrate with email or slack to automatically receive a report of Condition outcomes after a run finishes processing.

<figure><img src="../../../.gitbook/assets/Screen Shot 2022-11-29 at 12.37.30 PM.png" alt=""><figcaption></figcaption></figure>

### Examples&#x20;

<pre><code><strong>Example 1: Alert if over 50% of high DEP (>=0.7) data contains PII
</strong><strong>
</strong>    >>> c = Condition(
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
    >>> dq.register_run_report(conditions=[c])</code></pre>

<pre><code><strong>Example 2: Alert if at least 20% of the dataset has drifted (Inference DataFrames only)
</strong>
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
    >>> dq.register_run_report(conditions=[c])</code></pre>

⚡️ [Get started](build-your-own-conditions.md) building your own Reports with Galileo Conditions
