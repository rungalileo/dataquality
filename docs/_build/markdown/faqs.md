---
description: You have questions, we have (some) answers!
---

# 🙋 FAQs



* ****[**How do I install the Galileo Python client?**](faqs.md#q-how-do-i-install-the-galileo-python-client)****
* #### ****[**I'm seeing errors importing dataquality in jupyter/google colab**](faqs.md#q-im-seeing-errors-importing-dataquality-in-jupyter-google-colab)****
* #### ****[**My run finished, but there's no data in the console! What went wrong?**](faqs.md#q-my-run-finished-but-theres-no-data-in-the-console-what-went-wrong)****
* #### [Can I Log custom metadata to my dataset?](faqs.md#q-can-i-log-custom-metadata-to-my-dataset)
* #### [How do I disable Galileo logging during model training?](faqs.md#q-how-do-i-disable-galileo-logging-during-model-training)
* ****[**How do I load a Galileo exported file for re-training?**](faqs.md#q-how-do-i-load-a-galileo-exported-file-for-re-training)****
* ****[**How do I get my NER data into huggingface format?**](faqs.md#q-how-do-i-get-my-ner-data-into-huggingface-format)****
* ****[**My spans JSON column for my NER data can't be loaded with json.loads**](faqs.md#q-my-spansjson-column-for-my-ner-data-cant-be-loaded-with-json.loads)****
* ****[**Galileo marked an incorrect span as a span shift error, but it looks like a wront tag error. What's going on?**](faqs.md#q-galileo-marked-an-incorrect-span-as-a-span-shift-error-but-it-looks-like-a-wrong-tag-error.-whats)****
* ****[**What do you mean when you say the deployment logs are written to Google Cloud?**](faqs.md#q-what-do-you-mean-when-you-say-the-deployment-logs-are-written-to-google-cloud)****
* ****[**Does Galileo store data in the cloud?**](faqs.md#q-does-galileo-store-data-in-the-cloud)****
* ****[**Where are the client logs stored?**](faqs.md#q-where-are-the-client-logs-stored)****
* ****[**Do you offer air-gapped deployments?**](faqs.md#q-do-you-offer-air-gapped-deployments)****
* ****[**How do I contact Galileo?**](faqs.md#q-how-do-i-contact-galileo)****
* ****[**How do I convert my vaex dataframe to pandas when using dq.metrics.get\_dataframe?**](faqs.md#q-how-do-i-convert-my-vaex-dataframe-to-a-pandas-dataframe-when-using-the-dq.metrics.get\_dataframe)****
* ****[**Importing dataquality throws a permissions error \`PermissionError\`**](faqs.md#q-importing-dataquality-throws-a-permissions-error-permissionerror)****
* ****[**vaex-core fails to build with Python 3.10 on MacOs Monterey**](faqs.md#q-vaex-core-fails-to-build-with-python-3.10-on-macos-monterey)****

### Q: How do I install the Galileo Python client?

```
pip install dataquality
```

### Q: I'm seeing errors importing dataquality in Jupyter / Google Colab

Make sure you running at least `dataquality >= 0.8.6`\
The first thing to try in this case it to **restart your kernel**. Dataquality uses certain python packages that require your kernel to be restarted after installation. \
In Jupyter you can click "Kernel -> Restart"

![](<.gitbook/assets/image (14).png>)

In Colab you can click "Runtime -> Disconnect and delete runtime"

![](<.gitbook/assets/image (26).png>)

If you already had [vaex](https://github.com/vaexio) installed on your machine prior to installing `dataquality,` there is a known bug when upgrading. \
\
**Solution:** \
`pip uninstall -y vaex-core vaex-hdf5 && pip install --upgrade --force-reinstall dataquality` \
``**And then restart your jupyter/colab kernel**

### Q: My run finished, but there's no data in the console! What went wrong?

Make sure you ran `dq.finish()` after the run.

&#x20;t's possible that:&#x20;

* your run hasn't finished processing&#x20;
* you've logged some data incorrectly
* you may have found a bug (congrats!

First, to see what happened to your data, you can run\
`dq.wait_for_run()` \
(you can optionally pass in the project and run name, or the most recent will be used)

This function will wait for your run to finish processing. If it's completed, check the console again by refreshing.

If that shows an exception, your run failed to be processed. You can see the logs from your model training by running `dq.get_dq_log_file()` which will download and return the path to your logfile. That may indicate the issue. Feel free to reach out to us for more help!&#x20;

### Q: Can I log custom metadata to my dataset?

Yes (glad you asked)! You can attach any metadata fields you'd like to your original dataset, as long as they are primitive datatypes (numbers and strings).&#x20;

In all available logging functions for input data, you can attach custom metadata:

```python
df = pd.DataFrame(
    {
        "id": [0,1,2,3], 
        "text": ["sen 1","sen 2","sen 3","sen 4"], 
        "label": [0, 1, 1, 0],
        "customer_score": [0.66, 0.98, 0.12, 0.05],
        "sentiment": ["happy", "sad", "happy", "angry"]
    }
)

dq.log_dataset(df, meta=["customer_score", "sentiment"])
```

```python
texts = [
    "Text sample 1",
    "Text sample 2",
    "Text sample 3",
    "Text sample 4"
]
labels = ["B", "C", "A", "A"]
meta = {
    "sample_importance": ["high", "low", "low", "medium"]
    "quality_ranking": [9.7, 2.4, 5.5, 1.2]
}
ids = [0, 1, 2, 3]
split = "training"

dq.log_data_samples(texts=texts, labels=labels, ids=ids, meta=meta split=split)
```

This data will show up in the console under the column dropdown\
![](<.gitbook/assets/image (7).png>)

And you can see any performance metric grouped by your categorical metadata\
![](<.gitbook/assets/image (36).png>)

Lastly, once active, you can further filter your data by your metadata fields, helping find high-value cohorts\
![](<.gitbook/assets/image (4).png>)****



### Q: How do I disable Galileo logging during model training?

****

See [Disabling Galileo](python-library-api/disabling-galileo.md)

### Q: How do I load a Galileo exported file for re-training?

****

```python
from datasets import Dataset, dataset_dict
file_name_train = "exported_galileo_sample_file_train.parquet"
file_name_val = "exported_galileo_sample_file_val.parquet"
file_name_test = "exported_galileo_sample_file_test.parquet"
ds_train = Dataset.from_parquet(file_name_train)
ds_val = Dataset.from_parquet(file_name_val)
ds_test = Dataset.from_parquet(file_name_test)

ds_exported = dataset_dict.DatasetDict({"train": ds_train, "validation": ds_val, "test": ds_test})
labels = ds_new["train"]["ner_labels"][0]

tokenized_datasets = hf.tokenize_and_log_dataset(ds_exported, tokenizer, labels)
train_dataloader = hf.get_dataloader(tokenized_datasets["train"], collate_fn=data_collator, batch_size=MINIBATCH_SIZE, shuffle=True)
val_dataloader = hf.get_dataloader(tokenized_datasets["validation"], collate_fn=data_collator, batch_size=MINIBATCH_SIZE, shuffle=False)
test_dataloader = hf.get_dataloader(tokenized_datasets["test"], collate_fn=data_collator, batch_size=MINIBATCH_SIZE, shuffle=False)
```

### Q: How do I get my NER data into huggingface format?

****

```
import dataquality as dq
from datasets import Dataset

dq.login()
# A vaex dataframe
df = dq.metrics.get_dataframe(
    project_name, run_name, split, hf_format=True, tagging_schema="BIO"
)
df.export("data.parquet")
ds = Dataset.from_parquet("data.parquet")
```

### Q: My `spans` JSON column for my NER data can't be loaded with `json.loads`

&#x20;If you're seeing an error similar to:\
`JSONDecodeError: Expecting ',' delimiter: line 1 column 84 (char 83)`\
\
It's likely the case that you have some data in your `text` field that is not valid json (extra quotes `"` or `'`). Unfortunately, we cannot modify the content of your span text, but we can strip out the `text` field with some regex.\
\
Given a pandas dataframe `df` with column `spans` (from a Galileo export) you can replace\
`df["spans"] = df.apply(json.loads)`\
\
with (make sure to `import re`)\
`df["spans"] = df.apply(lambda row: json.loads(re.sub(r","text".}", "}", row)))`

### Q: Galileo marked an incorrect span as a span shift error, but it looks like a wrong tag error. What's going on?

Great observation! Let's take a real example below, from the WikiNER IT dataset. As you can see, the `Anemone apennina` clearly looks like a wrong tag error (correct span boundaries, incorrect class prediction), but is marked as a span shift.

![](<.gitbook/assets/image (23).png>)

We can further validate this with `dq.metrics.get_dataframe`. We can see that there are 2 spans with identical character boundaries, one with a label and one without (which is the prediction span).

![](<.gitbook/assets/image (6) (1).png>)&#x20;

So what is going on here? When Galileo computes error types for each span, they are computed at the _byte-pair (BPE)_ level using the span **token** indices, not **** the **character** indices. When looking at the console, however, you are seeing the **character** level indices, because that's much more intuitive view of your data. That conversion from **token** (fine-grained) to **character** (coarse-grained) level indices can cause index differences to overlap as a result of less-granular information.

We can again validate this with `dq.metrics` by looking at the raw data logged to Galileo. As we can see, at the **token** level, the span start and end indices do not align, and in fact overlap (ids 21948 and 21950), which is the reason for the span\_shift error 🤗

![](<.gitbook/assets/image (28).png>)

### Q: What do you mean when you say the deployment logs are written to Google Cloud?

We manage deployments and updates to the versions of services running in your cluster via Github Actions. Each deployment/update produces logs that go into a bucket on Galileo's cloud (GCP). During our private deployment process **** (for Enterprise users), we allow customers to provide us with their emails, so they can have access to these deployment logs.

### Q: Where are the client logs stored?

The client logs are stored in the home (\~) folder of the machine where the training occurs.

### Q: Does Galileo store data in the cloud?

For Enterprise Users, data does not leave the customer VPC/Data Center. For users of the Free version of our product, we store data and model outputs in secured servers in the cloud. We pride ourselves in taking data security very seriously.

### Q: Do you offer air-gapped deployments?

Yes, we do! Contact us to learn more.

### Q: How do I contact Galileo?

You can write us at team\[at]rungalileo.io

### Q: How do I convert my vaex dataframe to a pandas DataFrame when using the `dq.metrics.get_dataframe`

Simply add `dq.metrics.get_dataframe(...).to_pandas_df()`

### Q: **Importing dataquality throws a permissions error `PermissionError`**

Galileo creates a folder in your system's `HOME` directory.  If you are seeing a `PermissionsError` it means that your system does not have access to your current `HOME` directory. This may happen in an automated CI system like AWS Glue. To overcome this, simply change your `HOME` python Environment Variable to somewhere accessible. For example, the current directory you are in

```python
import os

# Set the HOME directory to the current working directory
os.environ["HOME"] = os.getcwd()
import dataquality as dq
```

This will only affect the current python runtime, it will not change your system's `HOME` directory. Because of that, if you run a new python script in this environment again, you will need to set the `HOME` variable in each new runtime.

### Q: vaex-core fails to build with Python 3.10 on MacOs Monterey

When installing dataquality with python 3.10 on MacOS Monterey you might encounter an issue when building vaex-core binaries. To fix any issues that come up, please follow the instructions in the failure output which may include running `xcodebuild -runFirstLaunch` and also allowing for any clang permission requests that pop up.

\
