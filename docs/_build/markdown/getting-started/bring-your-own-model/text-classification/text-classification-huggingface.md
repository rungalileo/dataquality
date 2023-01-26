# Huggingface 🤗

### Logging the Data Inputs

Log a human-readable version of your dataset. Galileo will join these samples with the model's outputs and present them in the Console. The data is recommended to be a Dataset class  from the datasets module. The id (index) column is needed to log our data and train the model.&#x20;

{% tabs %}
{% tab title="Huggingface" %}
<pre class="language-python"><code class="lang-python">import dataquality as dq
<strong>from datasets import load_dataset
</strong><strong>
</strong>dq.init(task_type="text_classification", # Change this based on your task type
        project_name="Sample_hf_project", 
        run_name="Sample_hf_run")

ds = load_dataset("emotion")
<strong># 🔭🌕 Galileo preprocessing (if the id column is not existing on the dataset)
</strong>ds = ds.map(lambda x,idx : {"id":idx}, with_indices=True)

train_dataset = ds["train"]
test_dataset = ds["test"]

# 🔭🌕 Galileo logging
dq.set_labels_for_run(train_dataset.features['label'].names)
dq.log_dataset(train_dataset, split="train")
dq.log_dataset(test_dataset, split="validation")
</code></pre>
{% endtab %}
{% endtabs %}

### Logging the Model Outputs

Log model outputs by creating the trainer and passing it into the dataquality watch functions

{% tabs %}
{% tab title="Huggingface" %}
```python
from dataquality.integrations.transformers_trainer import watch
from transformers import TrainingArguments, Trainer

args = TrainingArguments(
    ...
    )

trainer = Trainer(
    model,
    args,
    train_dataset=encoded_train_dataset,
    eval_dataset=encoded_test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
# 🔭🌕 Galileo logging
watch(trainer)
```
{% endtab %}
{% endtabs %}

### Training the Model

Now you are ready to train your model! Behind the scenes Galileo will track the different stages of training and will combine your model outputs with your logged input data.

{% tabs %}
{% tab title="Huggingface" %}
```python
...
trainer.train()    
...
dq.finish()  # 🔭🌕 This will wait until the run is processed by Galileo 
```
{% endtab %}
{% endtabs %}

### Example Notebooks

* [Text Classification Example Notebook](https://colab.research.google.com/github/rungalileo/examples/blob/v1/examples/text\_classification/Text\_Classification\_using\_Huggingface\_Trainer\_and\_%F0%9F%94%AD\_Galileo.ipynb)
* [Named Entity Recognition Example Notebook](https://colab.research.google.com/github/rungalileo/examples/blob/main/examples/named\_entity\_recognition/Named\_Entity\_Recognition\_with\_Pytorch\_and\_%F0%9F%94%AD\_Galileo.ipynb)
