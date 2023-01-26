---
description: Text Classification Inference
---

# Inference Run with Galileo: A Quick guide

{% hint style="info" %}
**\[**:medal: **Enterprise-Only] Running Inference is only available for Enterprise users for now.**&#x20;

Stay tuned for future announcements.
{% endhint %}

### Example Colab Notebooks

* [PyTorch](https://colab.research.google.com/drive/1L\_WLF86v1xxVJ9Fu-OC7ZuaOi03nsggQ#scrollTo=DB\_vQcupJ24T)

After building and training a model, inference allows us to run that model on unseen data, such as deploying that model in production. In text classification, given an unseen set of documents, the task is to predict (as correctly as possible) the class of that document based on the data seen during training.

```
input = "Perfectly works fine after 10 years, would highly recommend. Great buy!!"
# Unknown output label
model.predict(input) --> "positive review" 
```

### Logging the Data Inputs

Log your inference dataset. Galileo will join these samples with the model's outputs and present them in the Console. Note that unlike training, where ground truth labels are present for validation, during inference we assume that no ground truth labels exist.

{% tabs %}
{% tab title="PyTorch" %}
```python
import torch
import dataquality
import pandas as pd
from transformers import AutoTokenizer

class InferenceTextDataset(torch.utils.data.Dataset):
    def __init__(
        self, dataset: pd.DataFrame, inference_name: str
    ):
        self.dataset = dataset

        # 🔭🌕 Galileo logging
        # Note 1: this works seamlessly because self.dataset has text, label, and
        # id columns. See `help(dq.log_dataset)` for more info
        # Note 2: We can set the inference_name for our run
        dq.log_dataset(self.dataset, split="inference", inference_name=inference_name)

        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.encodings = tokenizer(
            self.dataset["text"].tolist(), truncation=True, padding=True
        )

    def __getitem__(self, idx):
        x = torch.tensor(self.encodings["input_ids"][idx])
        attention_mask = torch.tensor(self.encodings["attention_mask"][idx])

        return self.dataset["id"][idx], x, attention_mask

    def __len__(self):
        return len(self.dataset)
```
{% endtab %}
{% endtabs %}

### Logging the Inference Model Outputs

Log model outputs from within your model's forward function.&#x20;

{% tabs %}
{% tab title="PyTorch" %}
```python
import torch
import torch.nn.functional as F
from torch.nn import Linear
from transformers import AutoModel


class TextClassificationModel(torch.nn.Module):
    """Defines a Pytorch text classification bert based model."""

    def __init__(self, num_labels: int):
        super().__init__()
        self.feature_extractor = AutoModel.from_pretrained("distilbert-base-uncased")
        self.classifier = Linear(self.feature_extractor.config.hidden_size, num_labels)

    def forward(self, x, attention_mask, ids):
        """Model forward function."""
        encoded_layers = self.feature_extractor(
            input_ids=x, attention_mask=attention_mask
        ).last_hidden_state
        classification_embedding = encoded_layers[:, 0]
        logits = self.classifier(classification_embedding)

        # 🔭🌕 Galileo logging
        dq.log_model_outputs(
            embs=classification_embedding, logits=logits, ids=ids
        )

        return logits
```
{% endtab %}
{% endtabs %}

### Putting it all together

Login and initialize a _new_ project + run name _or_ one matching an existing training run (this will add inference to that training run in the console). Then, load and log your inference dataset; load a pre-trained model; set the split to inference and run your inference run; finally call `dq.finish()`!

Note: if you're extending a current training run, the `list_of_labels` logged for your dataset must match exactly that used during training.

{% tabs %}
{% tab title="PyTorch" %}
```python
import numpy as np
import io
import random
from smart_open import open as smart_open
import s3fs
import torch
import torch.nn.functional as F
import torchmetrics
from tqdm.notebook import tqdm

BATCH_SIZE = 32

# 🔭🌕 Galileo logging - initialize project/run name

dq.login()
dq.init(task_type="text_classification", project_name=project_name, run_name=run_name)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

inference_dataset = InferenceTextDataset(inference_df, inference_name="inference_run_1")

# 🔭🌕 Galileo logging
# Note: if you are adding the inference run to a previous
# training run, the labels and there order must match that used 
# in training. If you're logging inference in isolation then
# this order does not matter.
list_of_labels = ["labels", "ordered", "from", "trianing"] 
dq.set_labels_for_run(list_of_labels)

inference_dataloader = torch.utils.data.DataLoader(
        inference_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False
)

# Load your pre-trained model
model_path = "path/to/your/model.pt"
model = TextClassificationModel(num_labels=len(list_of_labels))
model.load_state_dict(torch.load(model_path))
model.to(device)

model.eval()

# 🔭🌕 Galileo logging - naming your inference run
inference_name = "inference_run_1"
dq.set_split("inference", inference_name)

for data in tqdm(inference_dataloader):
    x_idxs, x, attention_mask = data
    x = x.to(device)
    attention_mask = attention_mask.to(device)

    model(x, attention_mask, x_idxs)

print("Finished Inference")

# 🔭🌕 Galileo logging
dq.finish()

print("Finished uploading")
```
{% endtab %}
{% endtabs %}

To learn more about **Data Drift**, **Class Boundary Detection** or other Model Monitoring features, check out the [Galileo Product Features Guide](../../glossary/galileo-product-features/).
