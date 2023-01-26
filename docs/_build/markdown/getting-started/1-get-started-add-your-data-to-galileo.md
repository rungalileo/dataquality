---
description: The fastest way to find data errors in Galileo
---

# Add your data to Galileo

When focusing on data-centric techniques for modeling, we believe it is important to focus on the data while keeping the model static. To enable this rapid workflow, we suggest you use the `dq.auto` workflow:

After installing dataquality: `pip install dataquality`

You simply add your data and wait for the model to train under the hood, and for Galileo to process the data. This processing can take between 5-15 minutes, depending on how much data you have.&#x20;

`auto` will wait until Galileo is completely done processing your data. At that point, you can go to the Galileo Console and begin inspecting.

```python
import dataquality as dq

dq.auto(train_data=train_df, val_data=val_df, test_data=test_df)
```

There are 3 general ways to use `auto`

* Pass dataframes to `train_data`, `val_data` and `test_data` (pandas or huggingface)
* Pass paths to local files to `train_data`, `val_data` and `test_data`
* Pass a path to a huggingface Dataset to the `hf_data` parameter

`dq.auto` supports both Text Classification and Named Entity Recognition tasks, with Multi-Label support coming soon. `dq.auto` automatically determines the task type based off of the provided data schema.

To see the other available parameters as well as more usage examples, see `help(dq.auto)`

To learn more about how `dq.auto` works, and why we suggest this paradigm, see [DQ Auto](dq-auto.md)

#### **Looking to inspect your own model?**

Use `auto` if:

* You are looking to apply the most data-centric techniques to improve your data
* You don’t yet have a model to train
* You want to agnostically understand and fix your available training data

If you have a well-trained model and want to understand its performance on your data, or you are looking to deploy an existing model and monitor it with Galileo, please use our [custom framework integrations](byom-bring-your-own-model/).
