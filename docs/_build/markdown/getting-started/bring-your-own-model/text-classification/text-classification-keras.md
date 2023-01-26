# Keras

### Logging the Data Inputs

Log a human-readable version of your dataset. Galileo will join these samples with the model's outputs and present them in the Console.

{% tabs %}
{% tab title="Keras" %}
```python
import dataquality as dq

dq.init(task_type="text_classification", # Change this based on your task type
        project_name="example_keras_project",
        run_name="example_keras_run")

# 🔭🌕 Log the class labels in the order they are outputted by the model
labels_list = ["positive review", "negative review", "very positive review", "very negative review"]
dq.set_labels_for_run(list_of_labels)

# 🔭🌕 Log your pandas/huggingface/tf datasets to Galileo
dq.log_dataset(train_dataset, split="train")
dq.log_dataset(test_dataset, split="test")
```
{% endtab %}
{% endtabs %}

### Logging the Model Outputs

Add our logging layers to your Keras model's definition. This works with the functional or sequential syntax for defining models in Keras.

{% tabs %}
{% tab title="Keras" %}
```python
from dataquality.integrations.keras import DataQualityLoggingLayer

model = keras.Sequential(
    [
        DataQualityLoggingLayer("ids"), # 🌕🔭 
        ...
        DataQualityLoggingLayer("embs"), # 🌕🔭 
        ...
        DataQualityLoggingLayer("probs"), # 🌕🔭 
    ]
)

model.summary()
```
{% endtab %}
{% endtabs %}

### Training Loop Callback

Make sure to compile your model to run eagerly if it's not the default; add ids to your model's inputs; and add the Galileo callback to _auto-log_ the epochs and splits.&#x20;

{% tabs %}
{% tab title="Keras" %}
```python
from dataquality.integrations.keras import add_ids_to_numpy_arr, DataQualityCallback

x_train = add_ids_to_numpy_arr(x_train, train_ids) # 🌕🔭 ids from dataset logging

model.compile(..., run_eagerly=True)

model.fit(x_train, y_train, ..., 
    callbacks=[ DataQualityCallback() ]) # 🌕🔭 

dq.finish()  # 🔭🌕 This will wait until the run is processed by Galileo 
```
{% endtab %}
{% endtabs %}

### Example Notebooks&#x20;

* [Text Classification Example Notebook](https://colab.research.google.com/github/rungalileo/examples/blob/v1/examples/text\_classification/Text\_Classification\_using\_Keras\_and\_%F0%9F%94%AD\_Galileo.ipynb)
