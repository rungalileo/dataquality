---
description: The fastest way to improve your data
---

# DQ Auto

{% hint style="info" %}
To get started with auto instantly, see [Getting Started](1-get-started-add-your-data-to-galileo.md)
{% endhint %}

Welcome to `auto`, your newest superpower in the world of Machine Learning!

We know now that **more** data isn’t the answer, **better** data is. But how do you find that data? We already know the answer to that: ✨Galileo✨

But how do you get started now, and iterate quickly with _**data-centric**_ techniques?

Enter: `dq.auto` the secret sauce to instant data insights. We handle the training, you focus on the data.

### What is auto?

`dq.auto` is a helper function to train the most cutting-edge transformer (or any of your choosing from HuggingFace) on your dataset so it can be processed by Galileo. You provide the data, let Galileo train the model, and you’re off to the races.

The goal of this tool, and Galileo at large, is to build a data-centric view of machine learning. Keep your model static and iterate on the dataset until it’s well-formed and well-representative of your problem space. This is the path to robust and stable ML models.

### What is auto _not?_

`auto` is _**not**_ an AutoML tool. It will not perform hyperparameter tuning, and will not search through a gallery of models to optimize every percentage of f1.

In fact, `auto` is quite the opposite. It intentionally keeps the model static, forcing you to understand and fix your data to improve performance.

### Why?

It turns out that in many (most) cases, **you don’t need to train your own model to find data insights**. In fact, you often don’t need to build your own custom model at all! [HuggingFace](https://huggingface.co/), and in particular [transformers](https://huggingface.co/docs/transformers/index), has brought the most cutting-edge deep learning algorithms straight to your fingertips, allowing you to leverage the best research has to offer in 1 line of code.

Transformer models have consistently outperformed their predecessors, and HuggingFace is constantly updating their fleet of _free_ models for anyone to download.

{% hint style="success" %}
So if you don’t _need_ to build a custom model anymore, why not let Galileo do it for you?
{% endhint %}

### Get Started

Simply install: `pip install --upgrade dataquality`

and use!

```python
import dataquality as dq

# Get insights on the official 'emotion' dataset
dq.auto(hf_data="emotion")
```

You can also provide data as files or pandas dataframes

```python
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
import dataquality as dq

# Load the newsgroups dataset from sklearn
newsgroups_train = fetch_20newsgroups(subset='train')
newsgroups_test = fetch_20newsgroups(subset='test')
# Convert to pandas dataframes
df_train = pd.DataFrame({"text": newsgroups_train.data, "label": newsgroups_train.target})
df_test = pd.DataFrame({"text": newsgroups_test.data, "label": newsgroups_test.target})

dq.auto(
     train_data=df_train, 
     test_data=df_test, 
     labels=newsgroups_train.target_names,
     project_name="newsgroups_work", 
     run_name="run_1_raw_data"
)
```

`dq.auto` works for:

* Text Classification datasets (given columns `text` and `label`). [Trec6 Example.](https://huggingface.co/datasets/rungalileo/trec6)
* NER datasets (give columns `tokens` and `tags` or `ner_tags`). [MIT\_movies Example.](https://huggingface.co/datasets/rungalileo/mit\_movies)

`auto` will automatically figure out your task and start the process for you.

For more docs and examples, see `help(dq.auto)` in your notebook! Happy data fixing 🚀
