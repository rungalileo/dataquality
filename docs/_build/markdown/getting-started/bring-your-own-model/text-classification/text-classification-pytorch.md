# PyTorch

### Logging the Data Inputs

Log a human-readable version of your dataset. Galileo will join these samples with the model's outputs and present them in the Console.

{% tabs %}
{% tab title="PyTorch" %}
```python
import dataquality as dq

dq.init(task_type="text_classification", # Change this based on your task type
        project_name="Sample_torch_project",
        run_name="Sample_torch_run")

# Log the class labels in the order they are outputted by the model
labels_list = ["positive review", "negative review", "very positive review", "very negative review"]
dq.set_labels_for_run(list_of_labels)

# 🔭🌕 Log your pandas/huggingface/torch datasets to Galileo
dq.log_dataset(train_dataset, split="train")
dq.log_dataset(test_dataset, split="test")
```
{% endtab %}
{% endtabs %}

### Logging the Model Outputs

Log model outputs from your PyTorch model's forward function.&#x20;

{% hint style="info" %}
Your model must be defined in the torch model-subclass-style and be executing eagerly.&#x20;
{% endhint %}

Dataquality provides a simple solution to hook into your model with a few lines of code. To log model outputs in PyTorch you provide the **model** and the **dataloaders** to the **watch** function. This will hook into the model and extract the logits and embeddings. It registers a forward hook under the hood and requires to run serial (one worker). For multiple workers look into [advanced logging](text-classification-pytorch.md#pytorch-logging-in-the-forward-function-of-the-model).

{% tabs %}
{% tab title="PyTorch" %}
### PyTorch logging by passing the model to the watch function

```python
# Import our integration for pytorch
from dataquality.integrations.torch import watch
from torch.utils.data import DataLoader
BATCH_SIZE = 64
# Using the dataloader from PyTorch is required
train_dataloader = DataLoader(encoded_train_dataset,
                              batch_size=BATCH_SIZE)
test_dataloader = DataLoader(encoded_test_dataset,
                                batch_size=BATCH_SIZE)
# Hook into the training process by providing the model
# and the dataloaders to the dq watch function             
# 🔭🌕 Logging the dataset with Galileo
watch(model, [train_dataloader, test_dataloader])
```
{% endtab %}
{% endtabs %}



### Training the Model

Now you are ready to train your model! Log where you are within the training pipeline (epoch and current split) and behind the scenes Galileo will track the different stages of training and will combine your model outputs with your logged input data.

{% tabs %}
{% tab title="PyTorch" %}
```python
...

for epoch in range(epochs):
    # 🔭🌕 Galileo logging
    dq.set_epoch(epoch)
    
    # 🔭🌕 Galileo logging training
    dq.set_split("train")
    train_epoch(...)
    
    # 🔭🌕 Galileo logging evaluation
    dq.set_split("test")
    evaluate_model(...)
...
dq.finish()  # 🔭🌕 This will wait until the run is processed by Galileo 
```
{% endtab %}
{% endtabs %}

### Advanced PyTorch logging in the forward function of the model

If your model is supposed to run on multiple GPUs or with multiple workers, we recommend the advanced implementation, where you log model outputs from your PyTorch model's forward function. Note: Your model must be defined in the PyTorch model-subclass-style and be executing eagerly. The advanced solution does not need the model to be watched. Example [Colab Notebook](https://colab.research.google.com/github/rungalileo/examples/blob/v1/examples/text\_classification/Text\_Classification\_using\_PyTorch\_and\_%F0%9F%94%AD\_Galileo.ipynb).

{% tabs %}
{% tab title="PyTorch" %}
### PyTorch logging in the forward function of the model

```python
import torch
import dataquality as dq

class TextClassificationModel(torch.nn.Module):
    """Defines a PyTorch text classification model."""
    ...

    def forward(self, input_ids, attention_mask, ids):
        """Model forward function."""
        ...
        # classification_embedding has shape - [batch x emb_dim]
        # Logits has shape - [batch x num_classes]
        # Generally we select the [CLS] token for classification embedding
        # for example to remove the [CLS] token:
        # classification_embedding = encoded_layers[:, 0]
        # 🔭🌕 Galileo logging
        dq.log_model_outputs(
            embs=classification_embedding,
            logits=logits,
            ids=ids
        )

        return logits
```
{% endtab %}
{% endtabs %}

### Example Notebooks&#x20;

* [Text Classification Example Notebook](https://colab.research.google.com/github/rungalileo/examples/blob/v1/examples/text\_classification/Text\_Classification\_using\_PyTorch\_and\_%F0%9F%94%AD\_Galileo\_Simple.ipynb)
* [Named Entity Recognition Example Notebook](https://colab.research.google.com/github/rungalileo/examples/blob/main/examples/named\_entity\_recognition/Named\_Entity\_Recognition\_with\_Pytorch\_and\_%F0%9F%94%AD\_Galileo.ipynb)
* [Multi-Label Text Classification Example Notebook](https://colab.research.google.com/github/rungalileo/examples/blob/v1/examples/multi\_label\_text\_classification/Multi\_Label\_Text\_Classification\_using\_PyTorch\_and\_%F0%9F%94%AD\_Galileo\_Simple.ipynb)
* [Natural Language Inference Example Notebook](https://colab.research.google.com/github/rungalileo/examples/blob/main/examples/natural\_language\_inference/Natural\_Language\_Inference\_using\_Pytorch\_and\_%F0%9F%94%AD\_Galileo.ipynb)
