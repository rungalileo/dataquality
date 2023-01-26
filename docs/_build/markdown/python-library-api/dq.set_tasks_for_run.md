---
description: A multi-label text classification function
---

# dq.set\_tasks\_for\_run

```python
def set_tasks_for_run(tasks: List[str], binary: bool = False) -> None:
```

Sets the task names for the run (multi-label case only). Sets the order of the tasks with respect to the model's probability vectors.

| Arguments | Text                                                                                                                                                                                                                                                                                                                                                                     |
| --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `tasks`   | <p></p><p>An ordered list of tasks ie <code>['happiness','anger','sadness']</code>. This order MUST match the order of probability vectors for each task that the model outputs. </p><p></p><p>This also must match the order of the labels logged in set_labels_for_run (meaning that the first list of labels must be the labels of the first task passed in here)</p> |
| `binary`  | Whether this is a binary multi label run. If `True`, tasks will also be set as your labels, and you should NOT call `dq.set_labels_for_run` it will be handled for you. Default `False`                                                                                                                                                                                  |

****
