# dq.log\_data\_samples

Equivalent to calling `dq.log_data_sample` multiple times. Usage depends on task type.

### Text Classification

```python
def log_data_samples(
    *, 
    ids: List[Union[int, str]],
    texts: List[str],
    labels: Optional[List[str]] = None,
    split: Optional[Split] = None,
    inference_name: Optional[str] = None,
    meta: Optional[Dict[str, List[Union[str, float, int]]]] = None,
) -> None:
```

| Arguments        | Text                                                                                                          |
| ---------------- | ------------------------------------------------------------------------------------------------------------- |
| `*`              | All parameters to this function must be keyword arguments                                                     |
| `ids`            | Samples' ids. Need to match ids in the model's output.                                                        |
| `texts`          | Text samples.                                                                                                 |
| `labels`         | Labels for text samples. Required if not an inference split.                                                  |
| `split`          | `train`/`test`/`validation`/`inference`. Can be set here or via `dq.set_split`.                               |
| `inference_name` | If logging inference data, a name for this inference data is required. Can be set here or via `dq.set_split`. |
| `meta`           | Metadata for the text sample. Format is the `{"metadata_field_name": metadata_field_value}`                   |

**Examples:**

```python
import dataquality as dq

dq.init(task_type="text_classification")
dq.log_data_samples(
    ids=[0, 1, 2, 3], 
    texts=[
        "Text sample 1",
        "Text sample 2",
        "Text sample 3",
        "Text sample 4"
    ], 
    labels=["B", "C", "A", "A"], 
    split="training")
```



### Named Entity Recognition

```python
def log_data_samples(
    *,
    texts: List[str],
    ids: List[int],
    text_token_indices: List[List[Tuple[int, int]]] = None,
    gold_spans: List[List[Dict]] = None,
    split: Optional[Split] = None,
    meta: Optional[Dict[str, List[Union[str, float, int]]]] = None,
) -> None:
```

| Arguments            | Text                                                                                                                                                                                                              |
| -------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `*`                  | All parameters to this function must be keyword arguments                                                                                                                                                         |
| `ids`                | The samples' IDs. Need to match ids in the model's output.                                                                                                                                                        |
| `texts`              | Text samples.                                                                                                                                                                                                     |
| `text_token_indices` | Token boundaries of each text sample, 1 list per sample. Used to convert the gold\_spans into token level spans internally. `t[0]` indicates the start index of the span and `t[1]` is the end index (exclusive). |
| `gold_spans`         | The model-level gold spans over the char index for each text sample. A `List[Dict]` per text sample. `start`, `end`, `label` are the required keys.                                                               |
| `split`              | `train`/`test`/`validation`/`inference`. Can be set here or via `dq.set_split`.                                                                                                                                   |
| `meta`               | Metadata for each text sample. Format is the `{"metadata_field_name": [metadata value per sample]}`                                                                                                               |

**Examples:**

```python
import dataquality as dq

dq.init(task_type="text_ner")
dq.log_data_samples(
    ids=[0, 1], 
    texts=[
            "The president is Joe Biden",
            "Joe Biden addressed the United States on Monday"
        ], 
    text_token_indices=[
        [(0, 3), (4, 13), (14, 16), (17, 20), (21, 27), (21, 27)],
        [(0, 3), (4, 9), (10, 19), (20, 23), (24, 30), (31, 37), (38, 40), (41, 47)]
    ],
    gold_spans=[
            [
                {"start":17, "end":27, "label":"person"}  # "Joe Biden"
            ],
            [
                {"start":0, "end":10, "label":"person"},    # "Joe Biden"
                {"start":30, "end":41, "label":"location"}  # "United States"
            ]
        ],
    split="training",
    meta={"Job Title": ["president", "president"]}
)
```
