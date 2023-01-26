# Spacy

{% hint style="success" %}
Spacy only works for Named Entity Recognition tasks.
{% endhint %}

### Initializing Run

{% tabs %}
{% tab title="Spacy" %}
```python
import dataquality as dq

# 🔭🌕 Galileo logging - initialize project/run name
dq.login()
dq.init(task_type="text_ner", project_name="example_project", run_name="example_run")
```
{% endtab %}
{% endtabs %}

### Logging the Data Inputs

Log a human-readable version of your dataset. Galileo will join these samples with the model's outputs and present them in the Console.

{% tabs %}
{% tab title="Spacy" %}
```python
from dataquality.integrations.spacy import log_input_examples

# Create Spacy examples from your data
# type(train_examples[0]) == spacy.training.example.Example

# 🔭🌕 Galileo logging
log_input_examples(train_examples, "training")
```
{% endtab %}
{% endtabs %}

### Watch nlp

Initialize your model, then call our watch function to wrap it and auto-log.&#x20;

`watch(nlp)` wraps the `spacy.Language` with some Galileo logging code to instrument the necessary metrics from the model.

{% tabs %}
{% tab title="Spacy" %}
```python
from dataquality.integrations.spacy import watch

optimizer = nlp.initialize(lambda: train_examples+test_examples)
# 🔭🌕 Galileo wrapper
watch(nlp)
```
{% endtab %}
{% endtabs %}

### Training Loop

Now you are ready to train your model! Log where you are within the training pipeline (epoch and current split) and behind the scenes Galileo will track the different stages of training and will combine your model outputs with your logged input data.

{% tabs %}
{% tab title="Spacy" %}
```python
...

for epoch in range(num_epochs):
    # 🔭🌕 Galileo logging
    dq.set_epoch(epoch)
  
    # 🔭🌕 Galileo logging
    # Training epoch
    dq.set_split("training") # 🔭🌕
    nlp.update(...)
  
    # 🔭🌕 Galileo logging
    # Evaluation
    dq.set_split("test") # 🔭🌕
    nlp.evaluate(...)

...
```
{% endtab %}
{% endtabs %}



### Uploading to Galileo

To finish, simply call `dq.finish` and your data will be uploaded and processed by the Galileo API server. This may take a few minutes, depending on the size of your dataset.&#x20;

{% tabs %}
{% tab title="Spacy" %}
```python
dq.finish()  # 🔭🌕 This will wait until the run is processed by Galileo 
```
{% endtab %}
{% endtabs %}

### Example Notebooks&#x20;

* [Named Entity Recognition Example Notebook](https://colab.research.google.com/github/rungalileo/examples/blob/v1/examples/named\_entity\_recognition/Named\_Entity\_Recognition\_with\_SpaCy\_and\_%F0%9F%94%AD\_Galileo.ipynb)
