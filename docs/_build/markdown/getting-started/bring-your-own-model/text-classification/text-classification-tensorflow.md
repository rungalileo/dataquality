# TensorFlow

### Logging the Data Inputs

Log a human-readable version of your dataset. Galileo will join these samples with the model's outputs and present them in the Console.

{% tabs %}
{% tab title="TensorFlow" %}
```python
import dataquality as dq

dq.init(task_type="text_classification", # Change this based on your task type
        project_name="example_tf_project", 
        run_name="example_tf_run")

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

Log model outputs from your TensorFlow model's forward function.

{% hint style="info" %}
Your model must be defined in the TF model-subclass-style and be executing eagerly.&#x20;
{% endhint %}

{% tabs %}
{% tab title="TensorFlow" %}
```python
import tensorflow as tf

class TextClassificationModel(tf.keras.Model):
    """Defines a TensorFlow text classification model."""
    ...

    def call(self, x, ids):
        """Model forward function."""
        ...
        # classification_embedding has shape - [batch x emb_dim]
        # Logits has shape - [batch x num_classes] 
        # Generally we select the [CLS] token for classification embedding

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

### Training the Model

Now you are ready to train your model! Log where you are within the training pipeline (epoch and current split) and behind the scenes Galileo will track the different stages of training and will combine your model outputs with your logged input data.

{% tabs %}
{% tab title="undefined" %}

{% endtab %}
{% endtabs %}

### Example Notebooks&#x20;

* [Text Classification Example Notebook](https://colab.research.google.com/github/rungalileo/examples/blob/v1/examples/text\_classification/Text\_Classification\_using\_Tensorflow\_and\_%F0%9F%94%AD\_Galileo.ipynb)
* [Named Entity Recognition Example Notebook](https://colab.research.google.com/github/rungalileo/examples/blob/main/examples/named\_entity\_recognition/Named\_Entity\_Recognition\_with\_Tensorflow\_and\_%F0%9F%94%AD\_Galileo.ipynb)
* [Multi-Label Text Classification Example Notebook](https://colab.research.google.com/github/rungalileo/examples/blob/v1/examples/multi\_label\_text\_classification/Multi\_Label\_Text\_Classification\_using\_TensorFlow\_and\_%F0%9F%94%AD\_Galileo.ipynb)
* [Natural Language Inference Example Notebook](https://colab.research.google.com/github/rungalileo/examples/blob/main/examples/natural\_language\_inference/Natural\_Language\_Inference\_using\_TensorFlow\_and\_%F0%9F%94%AD\_Galileo.ipynb)
