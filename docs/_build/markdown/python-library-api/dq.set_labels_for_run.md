# dq.set\_labels\_for\_run

```python
def set_labels_for_run(labels: Union[List[str], List[List[str]]]) -> None:
```

Sets the ordering of the labels for the model's output logits or probabilities.

| Arguments | Text                                                                                                                                                                                                                                                                                                                                                                                                       |
| --------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `labels`  | <p></p><p>An ordered list of labels ie <code>['dog','cat','fish']</code>. This order MUST match the order of probabilities that the model outputs. </p><p></p><p>If this is a multi-label task, then labels are a list of lists where each inner list indicates the label for the given task. The outer order (order of the tasks) must match the task-order of the task-probabilities logged as well.</p> |

****

**Examples:**

Typically used after logging your dataset.

```python
import dataquality as dq

dq.set_labels_for_run(['dog','cat','fish'])

# ... Later on the model's output logits must be a vector of length 3 in the above order
```
