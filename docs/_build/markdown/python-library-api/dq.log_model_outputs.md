# dq.log\_model\_outputs

Logs a batch of model outputs for model during a forward pass.&#x20;

### Text Classification

```python
def log_model_outputs(
    ids: Union[List, np.ndarray],
    embs: Union[List, np.ndarray],
    logits: Union[List, np.ndarray],
    split: Optional[str] = None,
    epoch: Optional[int] = None,
    inference_name: str = None,
) -> None:
```

| Arguments        | Text                                                                                                    |
| ---------------- | ------------------------------------------------------------------------------------------------------- |
| `ids`            | The ids for each sample. Must match input ids of logged samples                                         |
| `embs`           | The embeddings for each sample in the batch.                                                            |
| `logits`         | The logits for each sample in the batch.                                                                |
| `split`          | The current split. Must be set either here or via `dq.set_split`.                                       |
| `epoch`          | The current epoch. Must be set either here or via `dq.set_epoch`                                        |
| `inference_name` | Inference name indicator for this inference split. If logging for an inference split, this is required. |

**Examples:**

```python
import dataquality as dq
import numpy as np

# ... within your model's forward function
dq.set_epoch(0)
dq.set_split("train")

embs: np.ndarray = np.random.rand(4, 768)  # 4 samples, embedding dim 768
logits: np.ndarray = np.random.rand(4, 3)  # 4 samples, 3 classes
ids: List[int] = [0, 1, 2, 3]

dq.log_model_outputs(embs=embs, logits=logits, ids=ids)
```



### Named Entity Recognition

```python
def log_model_outputs(
    ids: List[np.ndarray],
    embs: List[np.ndarray],
    logits: List[np.ndarray],
    split: Optional[str] = None,
    epoch: Optional[int] = None,
    inference_name: str = None,
) -> None:
```

| Arguments        | Text                                                                                                                                                                                                                                                                                                                                   |
| ---------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ids`            | The ids for each sample. Must match input ids of logged samples                                                                                                                                                                                                                                                                        |
| `embs`           | Each `np.ndarray` represents all embeddings of a given sample. These embeddings are from the tokenized text, and will align with the tokens in the sample. If you have 12 samples in the dataset, with the first sample 20 tokens in length, and an embedding vector of size 768, `len(embs) == 12`, and `embs[0].shape == (20, 768)`. |
| `logits`         | The NER prediction logits from the model for each token. These outputs are from the tokenized text, and will align with the tokens in the sample. If you have 12 samples in the dataset, with the first sample 20 tokens in length, and the number of classes is 40, `len(logits) == 12`, and `logits.shape == (20, 40)`.              |
| `split`          | The current split. Must be set either here or via `dq.set_split`.                                                                                                                                                                                                                                                                      |
| `epoch`          | The current epoch. Must be set either here or via `dq.set_epoch`.                                                                                                                                                                                                                                                                      |
| `inference_name` | Inference name indicator for this inference split. If logging for an inference split, this is required.                                                                                                                                                                                                                                |

**Examples:**

```python
import dataquality as dq
import numpy as np

# ... within your model's forward function
dq.set_epoch(0)
dq.set_split("train")

logits =
    [np.array([model(the), model(president), model(is), model(joe),
    model(bi), model(##den), model(<pad>), model(<pad>), model(<pad>)]),
    np.array([model(joe), model(bi), model(##den), model(addressed),
    model(the), model(united), model(states), model(on), model(monday)])]

embs =
    [np.array([emb(the), emb(president), emb(is), emb(joe),
    emb(bi), emb(##den), emb(<pad>), emb(<pad>), emb(<pad>)]),
    np.array([emb(joe), emb(bi), emb(##den), emb(addressed),
    emb(the), emb(united), emb(states), emb(on), emb(monday)])]

epoch = 0
ids = [0, 1]  # Must match the data input IDs
split = "training"
dataquality.log_model_outputs(
    embs=embs, logits=logits, ids=ids, split=split, epoch=epoch
)
```
