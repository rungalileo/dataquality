# dq.log\_dataset

Log a dataset/iterable of input samples. Usage depends on task type.

### Text Classification

```python
def log_dataset(
    dataset: Union[Iterable, pd.DataFrame, DataFrame],
    *,
    id: Union[str, int] = "id",
    text: Union[str, int] = "text",
    label: Optional[Union[str, int]] = "label",
    split: Optional[Split] = None,
    inference_name: Optional[str] = None,
    meta: Optional[List[Union[str, int]]] = None,
) -> None:
```

| Arguments        | Text                                                                                                                                                                                                            |
| ---------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `dataset`        | The dataset to log. This can be a python iterable or Pandas/Vaex dataframe. If an iterable, it can be a list of elements that can be indexed into either via int index (tuple/list) or string/key index (dict). |
| `*`              | All other parameters (below this one) must be keyword arguments                                                                                                                                                 |
| `id`             | Indication of which column (or index) of the dataframe is the `id`. If the column in your dataframe holding the id is **not** called id, you can pass in its name here.                                         |
| `text`           | Indication of which column (or index) of the dataframe is the `text`. If the column in your dataframe holding the text is **not** called text, you can pass in its name here.                                   |
| `labels`         | Indication of which column (or index) of the dataframe is the `label`. If the column in your dataframe holding the label is **not** called label, you can pass in its name here.                                |
| `split`          | `train`/`test`/`validation`/`inference`. Can be set here or via `dq.set_split`.                                                                                                                                 |
| `inference_name` | If logging inference data, a name for this inference data is required. Can be set here or via `dq.set_split`.                                                                                                   |
| `meta`           | The keys/indexes of each metadata field. Consider a pandas dataframe, this would be the list of columns corresponding to each metadata field to log.                                                            |

**Examples:**

```python
import dataquality as dq
import pandas as pd

dq.init(task_type="text_classification")

dataset = pd.DataFrame({
    "id": [0, 1, 2, 3],
    "text": [
        "Text sample 1",
        "Text sample 2",
        "Text sample 3",
        "Text sample 4"
    ],
    "label": ["B", "C", "A", "A"]
})

# Because our columns match the defaults, we can log as-is
dq.log_dataset(dataset, split="training")
```

```python
import dataquality as dq
import pandas as pd

dq.init(task_type="text_classification")

dataset = pd.DataFrame({
    "unique_id": [0, 1, 2, 3],
    "sentences": [
        "Text sample 1",
        "Text sample 2",
        "Text sample 3",
        "Text sample 4"
    ],
    "gold": ["B", "C", "A", "A"]
})

dq.log_dataset(dataset, split="training", text="sentences", id="unique_id", label="gold")
```

### Named Entity Recognition

```python
def log_dataset(
    dataset: Union[Iterable, pd.DataFrame, DataFrame],
    *,
    id: Union[str, int] = "id",
    text: Union[str, int] = "text",
    text_token_indices: Union[str, int] = "text_token_indices",
    gold_spans: Union[str, int] = "gold_spans",
    split: Optional[Split] = None,
    meta: Optional[List[Union[str, int]]] = None,
) -> None:
```

| Arguments            | Text                                                                                                                                                                                                            |
| -------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `dataset`            | The dataset to log. This can be a python iterable or Pandas/Vaex dataframe. If an iterable, it can be a list of elements that can be indexed into either via int index (tuple/list) or string/key index (dict). |
| `*`                  | All other parameters (below this one) must be keyword arguments                                                                                                                                                 |
| `id`                 | The key/index of the id fields.                                                                                                                                                                                 |
| `text`               | The key/index of the text fields.                                                                                                                                                                               |
| `text_token_indices` | The key/index of the sample text\_token\_indices.                                                                                                                                                               |
| `gold_spans`         | The key/index of the sample gold\_spans.                                                                                                                                                                        |
| `split`              | `train`/`test`/`validation`/`inference`. Can be set here or via `dq.set_split`.                                                                                                                                 |
| `meta`               | The keys/indexes of each metadata field. Consider a pandas dataframe, this would be the list of columns corresponding to each metadata field to log.                                                            |

**Examples:**

```python
import dataquality as dq

dq.init(task_type="text_ner")
dataset = pd.DataFrame({
    "id": [0, 1], 
    "text": [
            "The president is Joe Biden",
            "Joe Biden addressed the United States on Monday"
        ], 
    "text_token_indices": [
        [(0, 3), (4, 13), (14, 16), (17, 20), (21, 27), (21, 27)],
        [(0, 3), (4, 9), (10, 19), (20, 23), (24, 30), (31, 37), (38, 40), (41, 47)]
    ],
    "gold_spans": [
            [
                {"start":17, "end":27, "label":"person"}  # "Joe Biden"
            ],
            [
                {"start":0, "end":10, "label":"person"},    # "Joe Biden"
                {"start":30, "end":41, "label":"location"}  # "United States"
            ]
        ],
    "Job Title": ["president", "president"]
})

dq.log_dataset(dataset, split="training", meta=["Job Title"])
```
