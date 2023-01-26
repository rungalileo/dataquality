# 🧪 \[Experimental] No-code Quickstart

This is an experimental feature and currently only supports Text Classification datasets.

{% embed url="https://colab.research.google.com/github/rungalileo/examples/blob/v1/examples/Galileo_Quickstart.ipynb" %}

Using the [Galileo Quickstart](https://colab.research.google.com/github/rungalileo/examples/blob/v1/examples/Galileo\_Quickstart.ipynb) notebook you can get insights on your data without writing any code. Click the play button, upload your dataset or choose an existing one,  and see your insights in the Galileo Console.

In the background, [Galileo Quickstart](https://colab.research.google.com/github/rungalileo/examples/blob/v1/examples/Galileo\_Quickstart.ipynb) is training an off-the-shelf model with your data. As a result, getting insights on your data can take a few minutes.

**Dataset format:**

* Your file must be in CSV format.
* Your dataset must contain a "text" and a "label" column&#x20;
* If your dataset contains a "split" column with "train" and "val" values, they will be used to determine the training and validation set for the model. Otherwise, a split will be automatically chosen for you.
* Other columns will be logged as metadata and can be viewed from the Galileo console. &#x20;
