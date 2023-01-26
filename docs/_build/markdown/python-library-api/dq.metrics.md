---
description: Helper functions to get your raw and processed data out of Galileo
---

# dq.metrics

Experiment to your heart's content using `dq.metrics` to access your raw probabilities, embeddings, and processed dataframes from Galileo. These helper functions make it easy to access the data you've been looking for your whole life.



### `get_dataframe`

Downloads the data processed by Galileo for a run/split as a Vaex dataframe.

Optionally include the raw logged embeddings, probabilities, or text-token-indices (NER only)

{% hint style="info" %}
**Special note for NER:** By default, the data will be downloaded at a sample level (1 row per sample text), with spans for each sample in a `spans` column in a spacy-compatible JSON format. If include\_emb is True, the data will be expanded into span level (1 row per span, with sample text repeated for each span row), in order to join the span-level embeddings
{% endhint %}

```python
def get_dataframe(
    project_name: str,
    run_name: str,
    split: Split,
    inference_name: str = "",
    file_type: FileType = FileType.arrow,
    include_embs: bool = False,
    include_probs: bool = False,
    include_token_indices: bool = False,
    hf_format: bool = False,
    tagging_schema: Optional[TaggingSchema] = None,
    filter: Union[FilterParams, Dict] = None,
    as_pandas: bool = True,
) -> DataFrame:
```

****

**Example:**

| Arguments               | Text                                                                                                                                                                                                                                                                                                                                                                          |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `project_name`          | project to download data for                                                                                                                                                                                                                                                                                                                                                  |
| `run_name`              | run to download data for                                                                                                                                                                                                                                                                                                                                                      |
| `split`                 | split to download data for                                                                                                                                                                                                                                                                                                                                                    |
| `file_type`             | the file type to download the data as. Default arrow. It's suggested to leave the default as is.                                                                                                                                                                                                                                                                              |
| `include_embs`          | Whether to include the full logged embeddings in the data. If True for NER, the sample-level rows will be expanded to span-level rows in order to join the embeddings. Default False                                                                                                                                                                                          |
| `include_probs`         | Whether to include the full logged probabilities in the data. Not available for NER runs. Default False                                                                                                                                                                                                                                                                       |
| `include_token_indices` | (NER only) Whether to include logged text\_token\_indices in the data. Useful for reconstructing tokens for retraining                                                                                                                                                                                                                                                        |
| `hf_format`             | (NER only) Whether to export your data in a huggingface compatible format. This will return a dataframe with `text`, `tokens`, `ner_tags`, `ner_tags_readable`, and `ner_labels` which is a mapping from your `ner_tags` to your labels                                                                                                                                       |
| `tagging_schema`        | (NER only) if `hf_format` is `True`, this must be set. Must be one of `BIO`, `BIOES`, or `BILOU`                                                                                                                                                                                                                                                                              |
| `filter`                | Optional filter to provide to restrict the distribution to only to matching rows. See the FilterParams section below, or, in code, `help(dq.metrics.FilterParams)`                                                                                                                                                                                                            |
| `as_pandas`             | Whether to return the dataframe as a pandas df (or vaex if `False`) If you are having memory issues (the data is too large), set this to `False`, and vaex will memory map the data. If any columns returned are multi-dimensional (embeddings, probabilities etc), vaex will **always** be returned, because pandas cannot support multi-dimensional columns. Default `True` |



**Examples:**

```python
import dataquality as dq
from dataquality.schemas.metrics import FilterParams


project = "my_project"
run = "my_run"
split = "training"

df = dq.metrics.get_dataframe(project, run, split)
# This will be a vaex dataframe because embeddings are multi-dimensional
df_with_embs = dq.metrics.get_dataframe(project, run, split, include_embs=True)

# Filter dataframe
df_high_dep = dq.metrics.get_dataframe(
    project, run, split, filter={"data_error_potential_low": 0.9}
)
# Or use the FilterParams
df_high_dep = dq.metrics.get_dataframe(
    project, run, split, filter=FilterParams(data_error_potential_low=0.9)}
)

# NER only
# This df will be at the sample level
df_with_tokens = dq.metrics.get_dataframe(project, run, split, include_token_indices=True)
# This df will be expanded to the span level
df_with_embs_and_tokens = dq.metrics.get_dataframe(project, run, split, include_embs=True, include_token_indices=True)

# Get your data into a huggingface dataset!
hf_df = dq.metrics.get_dataframe(
    project, run, split, hf_format=True, tagging_schema="BIO"
)
hf_df.export("data.parquet")
from datasets import Dataset
ds = Dataset.from_parquet("data.parquet")
```

### FilterParams

When using the `get_dataframe` function, there is a parameter called `filter` which can be passed in as a dictionary, or as a `FilterParams` object.&#x20;

You can import the FilterParams using `dq.metrics.FilterParams` and run `help(dq.metrics.FilterParams)` to see all available filters.&#x20;

The list of currently available filters:

|                                                                              |                                                                                                                                                                                                                                                                                                                                                                                                                    |
| ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| <pre><code>ids: List[int] = []
</code></pre>                                 | Filter for specific IDs in the dataframe (span in NER)                                                                                                                                                                                                                                                                                                                                                             |
| <pre><code>similar_to: Optional[int] = None
</code></pre>                    | Provide an ID to run similarity search on                                                                                                                                                                                                                                                                                                                                                                          |
| <pre><code>num_similar_to: Optional[int] = None
</code></pre>                | If running similarity search, how many similar samples to get. More willl take longer                                                                                                                                                                                                                                                                                                                              |
| <pre><code>text_pat: Optional[StrictStr]
</code></pre>                       | Filter text samples by some text pattern                                                                                                                                                                                                                                                                                                                                                                           |
| <pre><code>regex: Optional[bool] = None
</code></pre>                        | If searching with text, whether to use regex                                                                                                                                                                                                                                                                                                                                                                       |
| <pre><code>data_error_potential_high: Optional[float] = None
</code></pre>   | Only samples with DEP <= this                                                                                                                                                                                                                                                                                                                                                                                      |
| <pre><code>data_error_potential_low: Optional[float] = None
</code></pre>    | Only samples with DEP >= this                                                                                                                                                                                                                                                                                                                                                                                      |
| <pre><code>misclassified_only: Optional[bool] = None
</code></pre>           | Only look at misclassified samples                                                                                                                                                                                                                                                                                                                                                                                 |
| <pre><code>gold_filter: Optional[List[StrictStr]] = None
</code></pre>       | Filter GT classes                                                                                                                                                                                                                                                                                                                                                                                                  |
| <pre><code>pred_filter: Optional[List[StrictStr]] = None
</code></pre>       | Filter prediction classes                                                                                                                                                                                                                                                                                                                                                                                          |
| <pre><code>class_filter: Optional[List[StrictStr]] = None
</code></pre>      | Filter for samples with these values as the GT OR prediction                                                                                                                                                                                                                                                                                                                                                       |
| <pre><code>meta_filter: Optional[List[MetaFilter]] = None 
</code></pre>     | Filter on particular metadata columns in the dataframe. see `help(dq.schemas.metrics.MetaFilter)`                                                                                                                                                                                                                                                                                                                  |
| <pre><code>inference_filter: Optional[InferenceFilter]
</code></pre>         | Specific fitlers related to inference data. See `help(dq.metrics.metrics.InferenceFilter)`                                                                                                                                                                                                                                                                                                                         |
| <pre><code>span_sample_ids: Optional[List[int]] = None
</code></pre>         | (NER only) filter for full samples by ID (will return all spans in those samples)                                                                                                                                                                                                                                                                                                                                  |
| <pre><code>span_text: Optional[str] = None
</code></pre>                     | (NER only) filter only on span text                                                                                                                                                                                                                                                                                                                                                                                |
| <pre><code>exclude_ids: List[int] = []
</code></pre>                         | Opposite of `ids` filter. Exclude the ids passed in (will apply to spans in NER)                                                                                                                                                                                                                                                                                                                                   |
| <pre><code>lasso: Optional[LassoSelection] = None
</code></pre>              | Related to making a lasso selection from the UI. See the `dq.schemas.metrics.LassoSelection` class                                                                                                                                                                                                                                                                                                                 |
| <pre><code>ikely_mislabeled: Optional[bool] = None
</code></pre>             | Filter for only likely\_mislabeled samples. False/None will return all samples                                                                                                                                                                                                                                                                                                                                     |
| <pre><code>likely_mislabeled_dep_percentile: Optional[int] = 0
</code></pre> | A percentile threshold for `likely_mislabeled`. This field (ranged 0-100) determines the precision of the likely\_mislabeled filter. The threshold is applied against the DEP distribution of the likely\_mislabeled samples. A threshold of 0 returns all, 100 returns 1 sample, and 50 will return the top 50% DEP samples that are likely\_mislabeled. Higher = more precision, lower = more recall. Default 0. |

### `get_edited_dataframe`

Downloads the data with all edits from the edits cart applied as a Vaex dataframe.

Optionally include the raw logged embeddings, probabilities, or text-token-indices (NER only)

{% hint style="info" %}
**Note:** This function has the identical syntax and signature as `get_dataframe` with the exception of no `filter` parameter. See `get_dataframe` above for a full list of parameters and examples. Anything passed into `get_dataframe` (besides `filter`) can be used exactly the same as `get_edited_dataframe`
{% endhint %}

```python
def get_edited_dataframe(
    project_name: str,
    run_name: str,
    split: Split,
    inference_name: str = "",
    file_type: FileType = FileType.arrow,
    include_embs: bool = False,
    include_probs: bool = False,
    include_token_indices: bool = False,
    hf_format: bool = False,
    tagging_schema: Optional[TaggingSchema] = None,
    as_pandas: bool = True
) -> DataFrame:
```

**Examples:**

```python
import dataquality as dq
from dataquality.schemas.metrics import FilterParams


project = "my_project"
run = "my_run"
split = "training"

# Export the edited dataframe with all edits from the edits cart
edited_df = dq.metrics.get_edited_dataframe(project, run, split)

# See the probabilities with your edited data (this will be a vaex dataframe)
edited_df = dq.metrics.get_edited_dataframe(project, run, split, include_probs=True)
```

### `get_run_summary`

Get the full run summary for a run/split. This provides:

* &#x20;overall metrics (weighted f1, recall, precision),&#x20;
* DEP distribution
* misclassified pairs
* top 50 samples sorted by DEP descending
* top DEP words (NER only)
* performance per task (Multi-label only)

```python
def get_run_summary(
    project_name: str,    
    run_name: str,    
    split: Split,    
    task: Optional[str] = None,    
    inference_name: Optional[str] = None,
    filter: Union[FilterParams, Dict] = None,
) -> Dict:
```

**Example:**

| Returns          | Text                                                                                                                                                           |
| ---------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Dict[str, Any]` | A dictionary of many different fields of interest, encompassing a "summary" of this run/split, including performance metrics, some samples, distributions etc. |

```python
import dataquality as dq
import pandas as pd


project = "my_project"
run = "my_run"
split = "training"

sumamry = dq.metrics.get_run_summary(
  project, run, split, 
)

print(summary)

# See summary for only misclassified samples
mis_sumamry = dq.metrics.get_run_summary(
  project, run, split, filter={"misclassified_only": True}
)
print(mis_sumamry)
```

### `get_metrics`

Get metrics for classes grouped by a particular categorical column (or ground truth or prediction)

```python
def get_metrics(
    project_name: str,
    run_name: str,
    split: Split,
    task: Optional[str] = None,
    inference_name: Optional[str] = None,
    category: str = "gold",
    filter: Union[FilterParams, Dict] = None,
) -> Dict[str, List]:
```

**Example:**

| Returns           | Text                                                                                                                                                                      |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Dict[str, List]` | A dictionary of keys -> list of values. The `labels` key is your x axis, and your other keys are potential y-axes (useful for plotting or inputting to a Pandas dataframe |

```python
import dataquality as dq
import pandas as pd


project = "my_project"
run = "my_run"
split = "training"

metrics = dq.metrics.get_metrics(
  project, run, split, category="galileo_language_id"
)

metrics_df = pd.DataFrame(metrics)

```



### `display_distribution`

Plots the distribution for a continuous column in your data. Defaults to Data Error Potential.

When plotting data error potential, the hard/easy DEP thresholds will be used for coloring. Otherwise no coloring is applied.

Plotly must be installed for this function to work.

```python
def display_distribution(
    project_name: str,
    run_name: str,
    split: Split,
    task: Optional[str] = None,
    inference_name: Optional[str] = None,
    column: str = "data_error_potential",
    filter: Union[FilterParams, Dict] = None,
) -> None:
```

**Examples:**

```python
import dataquality as dq


project = "my_project"
run = "my_run"
split = "training"

# Display DEP distribution colored by thresholds
dq.metrics.display_distribution(
  project, run, split
)

# Display text length distribution
dq.metrics.display_distribution(
  project, run, split, column="galileo_text_length"
)

# Display DEP distribution only for with gold/pred = class "APPLE"
dq.metrics.display_distribution(
  project, run, split, filter={"class_filter": ["APPLE"]}
)

```

### `get_epochs`

Returns the list of epochs logged for a run/split

```python
def get_epochs(
  project_name: str, run_name: str, split: Split
) -> List[int]:
```

**Examples:**

```python
import dataquality as dq


project = "my_project"
run = "my_run"
split = "training"

logged_epochs = dq.metrics.get_epochs(project, run, split)

```

### `get_embeddings`

Downloads the embeddings for a run/split at an epoch as a Vaex dataframe.

Optionally choose the epoch to get embeddings for, otherwise the latest epoch's embeddings will be chosen.

```python
def get_embeddings(
    project_name: str, run_name: str, split: Split, epoch: int = None
) -> DataFrame:
```

**Examples:**

```python
import dataquality as dq


project = "my_project"
run = "my_run"
split = "training"

latest_embs = dq.metrics.get_embeddings(project, run, split)
epochs = sorted(dq.metrics.get_epochs(project, run, split))
second_latest_embs = dq.metrics.get_embeddings(project, run, split, epoch=epochs[-2])

```

### `get_probabilities`

Downloads the probabilities for a run/split at an epoch as a Vaex dataframe.

Optionally choose the epoch to get probabilities for, otherwise the latest epoch's probabilities will be chosen.

```python
def get_probabilities(
    project_name: str, run_name: str, split: Split, epoch: int = None
) -> DataFrame:
```

**Examples:**

```python
import dataquality as dq


project = "my_project"
run = "my_run"
split = "training"

latest_probs = dq.metrics.get_probabilities(project, run, split)
epochs = sorted(dq.metrics.get_epochs(project, run, split))
second_latest_probs = dq.metrics.get_probabilities(project, run, split, epoch=epochs[-2])

```

### `get_raw_data`

Downloads the raw logged data for a run/split at an epoch as a Vaex dataframe.

Optionally choose the epoch to get probabilities for, otherwise the latest epoch's probabilities will be chosen.

For NER, this will download the text samples and text-token-indices

```python
def get_raw_data(
    project_name: str, run_name: str, split: Split, epoch: int = None
) -> DataFrame:
```

**Examples:**

```python
import dataquality as dq


project = "my_project"
run = "my_run"
split = "training"

df = dq.metrics.get_raw_data(project, run, split)

```

### `get_labels_for_run`

Gets labels for a given run. If multi-label, a task must be provided

```python
def get_label_for_run(
    project_name: str, run_name: str, task: Optional[str] = None
) -> List[str]:
```

**Examples:**

```python
import dataquality as dq


project = "my_project"
run = "my_run"

labels = dq.metrics.get_labels_for_run(project, run)

# for multi-label 
tasks = dq.metrics.get_tasks_for_run(project, run, split)
labels = dq.metrics.get_labels_for_run(project, run, tasks[0])

```

### `get_tasks_for_run`

Multi-label only. Gets tasks for a given run.

```python
def get_tasks_for_run(project_name: str, run_name: str) -> List[str]:
```

**Examples:**

```python
import dataquality as dq


project = "my_project"
run = "my_run"

tasks = dq.metrics.get_tasks_for_run(project, run, split)

```

### `get_xray_cards`

Get xray cards for a project/run/split

Xray cards are automatic insights calculated and provided by Galileo on your data

```python
def get_xray_cards(
    project_name: str, run_name: str, split: Split, inference_name: Optional[str] = None
) -> List[Dict[str, str]]:
```

**Examples:**

```python
import dataquality as dq


project = "my_project"
run = "my_run"
split = "training"

tasks = dq.metrics.get_xray_cards(project, run, split)

```
