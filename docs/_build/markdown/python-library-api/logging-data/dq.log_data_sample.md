# dq.log\_data\_sample

Logs a single data sample. Specific usage depends on the task type.

### Text Classification <a href="#text-classification" id="text-classification"></a>

```python
def log_data_sample(
    *,
    id: Union[int, str],
    text: str,
    label: Optional[str] = None,
    split: Optional[Split] = None,
    inference_name: Optional[str] = None,
    meta: Optional[Dict[str, Union[str, float, int]]] = None,
) -> None:
```

| Arguments        | Text                                                                                                          |
| ---------------- | ------------------------------------------------------------------------------------------------------------- |
| `*`              | All parameters to this function must be keyword arguments                                                     |
| `id`             | The sample's ID. Needs to match an id in the model's output.                                                  |
| `text`           | The text sample.                                                                                              |
| `label`          | Label for the sample. Required if not an inference split.                                                     |
| `split`          | `train`/`test`/`validation`/`inference`. Can be set here or via `dq.set_split`.                               |
| `inference_name` | If logging inference data, a name for this inference data is required. Can be set here or via `dq.set_split`. |
| `meta`           | Metadata for the text sample. Format is the `{"metadata_field_name": metadata_field_value}`                   |

**Examples:**

```python
import dataquality as dq

dq.init(task_type="text_classification")
dq.log_data_sample(id=0, text="some text here", label="class_0", split="training")
```



### Named Entity Recognition

```python
def log_data_sample(
    *, 
    id: Union[int, str],
    text: str,
    text_token_indices: List[Tuple[int, int]] = None,
    gold_spans: List[Dict] = None,
    split: Optional[Split] = None,
    meta: Optional[Dict[str, Union[str, float, int]]] = None,
) -> None:
```

| Arguments            | Text                                                                                                                                                                                      |
| -------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `*`                  | All parameters to this function must be keyword arguments                                                                                                                                 |
| `id`                 | The sample's ID. Needs to match an id in the model's output.                                                                                                                              |
| `text`               | The text sample.                                                                                                                                                                          |
| `text_token_indices` | Token boundaries of the text sample. Used to convert gold\_spans into token level spans internally. `t[0]` indicates the start index of the span and `t[1]` is the end index (exclusive). |
| `gold_spans`         | The model-level gold spans over the char index of the text sample. `start`, `end`, `label` are the required keys.                                                                         |
| `split`              | `train`/`test`/`validation`/`inference`. Can be set here or via `dq.set_split`.                                                                                                           |
| `meta`               | Metadata for the text sample. Format is the `{"metadata_field_name": metadata_field_value}`                                                                                               |

**Examples:**

```python
import dataquality as dq

dq.init(task_type="text_ner")
dq.log_data_sample(
    id=0, 
    text="The president is Joe Biden", 
    text_token_indices=[(0, 3), (4, 13), (14, 16), (17, 20), (21, 27), (21, 27)],
    gold_spans=[
                {"start":17, "end":27, "label":"person"}  # "Joe Biden"
            ],
    split="training",
    meta={"Job Title": "president"}
)
```

