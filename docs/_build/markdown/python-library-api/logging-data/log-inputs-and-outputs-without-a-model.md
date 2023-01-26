---
description: Manually upload your model data to Galileo
---

# Log inputs and outputs without a model

It may be the case that you've already trained your machine learning models, or don't want to integrate the `dataquality` client with your training loop. That's okay! You can use the client to directly upload your data to the Galileo server and inspect it in the console.

Below we have an example of generating some mock data, but you could replace these inputs with the outputs of your models and be on your way.

{% hint style="success" %}
This code is runnable as is
{% endhint %}

First, we'll define some functions for generating data for the purposes of the example (this is where you'd pass in your real ML data):

```python
from random import choice 
import numpy as np
from string import ascii_lowercase
import pandas as pd

labels = ["APPLE", "BANANA", "CARROT", "GRAPE", "BLUEBERRY"]

def generate_sentence(text_len=25) -> str:
    return "".join(choice(ascii_lowercase + " ") for i in range(text_len)) 

def load_data_samples(n) -> pd.DataFrame:
    df = dict(
        id=list(range(n)),
        label=[choice(labels) for _ in range(n)], 
        text=[generate_sentence() for _ in range(n)]
    )
    return pd.DataFrame(df)

def load_model_epoch_data(n) -> pd.DataFrame:
    data = dict(
        ids=list(range(n)),
        embs=np.random.rand(n, 700),
        logits=np.random.rand(n, len(labels))
    )
    return data

```

Now you can pass your data to the same log functions as you'd use during training

```python
import dataquality as dq 

dq.init(
    task_type="text_classification", 
    project_name="manual_upload", 
    run_name="first_run"
)
dq.set_labels_for_run(labels)

dq.set_split("training")
    
num_samples = 10_000

input_df = load_data_samples(num_samples)
dq.log_dataset(input_df)

# Log 3 epochs worth of data
for epoch in range(3):
    model_output = load_model_epoch_data(num_samples)
    dq.set_epoch(epoch)
    dq.log_model_outputs(**model_output)

dq.finish()
```
