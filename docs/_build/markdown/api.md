# API

dataquality


### _class_ AggregateFunction(value)
An enumeration.


### _class_ Condition(\*\*data)
Class for building custom conditions for data quality checks

After building a condition, call evaluate to determine the truthiness
of the condition against a given DataFrame.

With a bit of thought, complex and custom conditions can be built. To gain an
intuition for what can be accomplished, consider the following examples:


1. Is the average confidence less than 0.3?

    ```python
    >>> c = Condition(
    ...     agg=AggregateFunction.avg,
    ...     metric="confidence",
    ...     operator=Operator.lt,
    ...     threshold=0.3,
    ... )
    >>> c.evaluate(df)
    ```


2. Is the max DEP greater or equal to 0.45?

    ```python
    >>> c = Condition(
    ...     agg=AggregateFunction.max,
    ...     metric="data_error_potential",
    ...     operator=Operator.gte,
    ...     threshold=0.45,
    ... )
    >>> c.evaluate(df)
    ```

By adding filters, you can further narrow down the scope of the condition.
If the aggregate function is “pct”, you don’t need to specify a metric,

> as the filters will determine the percentage of data.

For example:


1. Alert if over 80% of the dataset has confidence under 0.1

    ```python
    >>> c = Condition(
    ...     operator=Operator.gt,
    ...     threshold=0.8,
    ...     agg=AggregateFunction.pct,
    ...     filters=[
    ...         ConditionFilter(
    ...             metric="confidence", operator=Operator.lt, value=0.1
    ...         ),
    ...     ],
    ... )
    >>> c.evaluate(df)
    ```


2. Alert if at least 20% of the dataset has drifted (Inference DataFrames only)

    ```python
    >>> c = Condition(
    ...     operator=Operator.gte,
    ...     threshold=0.2,
    ...     agg=AggregateFunction.pct,
    ...     filters=[
    ...         ConditionFilter(
    ...             metric="is_drifted", operator=Operator.eq, value=True
    ...         ),
    ...     ],
    ... )
    >>> c.evaluate(df)
    ```


3. Alert 5% or more of the dataset contains PII

    ```python
    >>> c = Condition(
    ...     operator=Operator.gte,
    ...     threshold=0.05,
    ...     agg=AggregateFunction.pct,
    ...     filters=[
    ...         ConditionFilter(
    ...             metric="galileo_pii", operator=Operator.neq, value="None"
    ...         ),
    ...     ],
    ... )
    >>> c.evaluate(df)
    ```

Complex conditions can be built when the filter has a different metric
than the metric used in the condition. For example:


1. Alert if the min confidence of drifted data is less than 0.15

    ```python
    >>> c = Condition(
    ...     agg=AggregateFunction.min,
    ...     metric="confidence",
    ...     operator=Operator.lt,
    ...     threshold=0.15,
    ...     filters=[
    ...         ConditionFilter(
    ...             metric="is_drifted", operator=Operator.eq, value=True
    ...         )
    ...     ],
    ... )
    >>> c.evaluate(df)
    ```


2. Alert if over 50% of high DEP (>=0.7) data contains PII

    ```python
    >>> c = Condition(
    ...     operator=Operator.gt,
    ...     threshold=0.5,
    ...     agg=AggregateFunction.pct,
    ...     filters=[
    ...         ConditionFilter(
    ...             metric="data_error_potential", operator=Operator.gte, value=0.7
    ...         ),
    ...         ConditionFilter(
    ...             metric="galileo_pii", operator=Operator.neq, value="None"
    ...         ),
    ...     ],
    ... )
    >>> c.evaluate(df)
    ```

You can also call conditions directly, which will assert its truth against a df
1. Assert that average confidence less than 0.3
>>> c = Condition(
…     agg=AggregateFunction.avg,
…     metric=”confidence”,
…     operator=Operator.lt,
…     threshold=0.3,
… )
>>> c(df)  # Will raise an AssertionError if False


* **Parameters**

    
    * **metric** – The DF column for evaluating the condition


    * **agg** – An aggregate function to apply to the metric


    * **operator** – The operator to use for comparing the agg to the threshold
    (e.g. “gt”, “lt”, “eq”, “neq”)


    * **threshold** – Threshold value for evaluating the condition


    * **filter** – Optional filter to apply to the DataFrame before evaluating the
    condition



### _class_ ConditionFilter(\*\*data)
Filter a dataframe based on the column value

Note that the column used for filtering is the same as the metric used

    in the condition.


* **Parameters**

    
    * **operator** – The operator to use for filtering (e.g. “gt”, “lt”, “eq”, “neq”)
    See Operator


    * **value** – The value to compare against



### _class_ Operator(value)
An enumeration.


### auto(hf_data=None, hf_inference_names=None, train_data=None, val_data=None, test_data=None, inference_data=None, max_padding_length=200, hf_model='distilbert-base-uncased', labels=None, project_name=None, run_name=None, wait=True, create_data_embs=False)
Automatically gets insights on a text classification or NER dataset

Given either a pandas dataframe, file_path, or huggingface dataset path, this
function will load the data, train a huggingface transformer model, and
provide Galileo insights via a link to the Galileo Console

One of hf_data, train_data should be provided. If neither of those are, a
demo dataset will be loaded by Galileo for training.


* **Parameters**

    
    * **hf_data** (`Union`[`DatasetDict`, `str`, `None`]) – Union[DatasetDict, str] Use this param if you have huggingface
    data in the hub or in memory. Otherwise see train_data, val_data,
    and test_data. If provided, train_data, val_data, and test_data are ignored.


    * **hf_inference_names** (`Optional`[`List`[`str`]]) – Use this param alongside hf_data if you have splits
    you’d like to consider as inference. A list of key names in hf_data
    to be run as inference runs after training. Any keys set must exist in hf_data


    * **train_data** (`Union`[`DataFrame`, `Dataset`, `str`, `None`]) – Optional training data to use. Can be one of
    \* Pandas dataframe
    \* Huggingface dataset
    \* Path to a local file
    \* Huggingface dataset hub path


    * **val_data** (`Union`[`DataFrame`, `Dataset`, `str`, `None`]) – Optional validation data to use. The validation data is what is
    used for the evaluation dataset in huggingface, and what is used for early
    stopping. If not provided, but test_data is, that will be used as the evaluation
    set. If neither val nor test are available, the train data will be randomly
    split 80/20 for use as evaluation data.
    Can be one of
    \* Pandas dataframe
    \* Huggingface dataset
    \* Path to a local file
    \* Huggingface dataset hub path


    * **test_data** (`Union`[`DataFrame`, `Dataset`, `str`, `None`]) – Optional test data to use. The test data, if provided with val,
    will be used after training is complete, as the held-out set. If no validation
    data is provided, this will instead be used as the evaluation set.
    Can be one of
    \* Pandas dataframe
    \* Huggingface dataset
    \* Path to a local file
    \* Huggingface dataset hub path


    * **inference_data** (`Optional`[`Dict`[`str`, `Union`[`DataFrame`, `Dataset`, `str`]]]) – User this param to include inference data alongside the
    train_data param. If you are passing data via the hf_data parameter, you
    should use the hf_inference_names param. Optional inference datasets to run
    with after training completes. The structure is a dictionary with the
    key being the inference name and the value one of
    \* Pandas dataframe
    \* Huggingface dataset
    \* Path to a local file
    \* Huggingface dataset hub path


    * **max_padding_length** (`int`) – The max length for padding the input text
    during tokenization. Default 200


    * **hf_model** (`str`) – The pretrained AutoModel from huggingface that will be used to
    tokenize and train on the provided data. Default distilbert-base-uncased


    * **labels** (`Optional`[`List`[`str`]]) – Optional list of labels for this dataset. If not provided, they
    will attempt to be extracted from the data


    * **project_name** (`Optional`[`str`]) – Optional project name. If not set, a random name will
    be generated


    * **run_name** (`Optional`[`str`]) – Optional run name for this data. If not set, a random name will
    be generated


    * **wait** (`bool`) – Whether to wait for Galileo to complete processing your run.
    Default True


    * **create_data_embs** (`bool`) – Whether to create data embeddings for this run. If True,
    Sentence-Transformers will be used to generate data embeddings for this dataset
    and uploaded with this run. You can access these embeddings via
    dq.metrics.get_data_embeddings in the emb column or
    dq.metrics.get_dataframe(…, include_data_embs=True) in the data_emb col
    Only available for TC currently. NER coming soon. Default False.



* **Return type**

    `None`


For text classification datasets, the only required columns are text and label

For NER, the required format is the huggingface standard format of tokens and
tags (or ner_tags).
See example: [https://huggingface.co/datasets/rungalileo/mit_movies](https://huggingface.co/datasets/rungalileo/mit_movies)

> MIT Movies dataset in huggingface format

> tokens                                              ner_tags
> [what, is, a, good, action, movie, that, is, r…       [0, 0, 0, 0, 7, 0, …
> [show, me, political, drama, movies, with, jef…       [0, 0, 7, 8, 0, 0, …
> [what, are, some, good, 1980, s, g, rated, mys…       [0, 0, 0, 0, 5, 6, …
> [list, a, crime, film, which, director, was, d…       [0, 0, 7, 0, 0, 0, …
> [is, there, a, thriller, movie, starring, al, …       [0, 0, 0, 7, 0, 0, …
> …                                               …                      …

To see auto insights on a random, pre-selected dataset, simply run


```
``
```



```
`
```

python

> import dataquality as dq

> dq.auto()



```
``
```



```
`
```


An example using auto with a hosted huggingface text classification dataset


```
``
```



```
`
```

python

> import dataquality as dq

> dq.auto(hf_data=”rungalileo/trec6”)



```
``
```



```
`
```


Similarly, for NER


```
``
```



```
`
```

python

> import dataquality as dq

> dq.auto(hf_data=”conll2003”)



```
``
```



```
`
```


An example using auto with sklearn data as pandas dataframes


```
``
```



```
`
```

python

> import dataquality as dq
> import pandas as pd
> from sklearn.datasets import fetch_20newsgroups

> # Load the newsgroups dataset from sklearn
> newsgroups_train = fetch_20newsgroups(subset=’train’)
> newsgroups_test = fetch_20newsgroups(subset=’test’)
> # Convert to pandas dataframes
> df_train = pd.DataFrame(

> > {“text”: newsgroups_train.data, “label”: newsgroups_train.target}

> )
> df_test = pd.DataFrame(

> > {“text”: newsgroups_test.data, “label”: newsgroups_test.target}

> )

> dq.auto(

>     train_data=df_train,
>     test_data=df_test,
>     labels=newsgroups_train.target_names,
>     project_name=”newsgroups_work”,
>     run_name=”run_1_raw_data”

> )



```
``
```



```
`
```


An example of using auto with a local CSV file with text and label columns


```
``
```



```
`
```

python
import dataquality as dq

dq.auto(

    train_data=”train.csv”,
    test_data=”test.csv”,
    project_name=”data_from_local”,
    run_name=”run_1_raw_data”

## )


### build_run_report(conditions, emails, project_id, run_id, link)
Build a run report and send it to the specified emails.


* **Return type**

    `None`



### configure(do_login=True)
[Not for cloud users] Update your active config with new information

You can use environment variables to set the config, or wait for prompts
Available environment variables to update:
\* GALILEO_CONSOLE_URL
\* GALILEO_USERNAME
\* GALILEO_PASSWORD


* **Return type**

    `None`



### docs()
Print the documentation for your specific input and output logging format

Based on your task_type, this will print the appropriate documentation


* **Return type**

    `None`



### finish(last_epoch=None, wait=True, create_data_embs=False)
Finishes the current run and invokes a job


* **Parameters**

    
    * **last_epoch** (`Optional`[`int`]) – If set, only epochs up to this value will be uploaded/processed
    This is inclusive, so setting last_epoch to 5 would upload epochs 0,1,2,3,4,5


    * **wait** (`bool`) – If true, after uploading the data, this will wait for the
    run to be processed by the Galileo server. If false, you can manually wait
    for the run by calling dq.wait_for_run() Default True


    * **create_data_embs** (`bool`) – If True, an off-the-shelf transformer will run on the raw
    text input to generate data-level embeddings. These will be available in the
    data view tab of the Galileo console. You can also access these embeddings
    via dq.metrics.get_data_embeddings()



* **Return type**

    `str`



### get_run_status(project_name=None, run_name=None)
Returns the latest job of a specified project run.
Defaults to the current run if project_name and run_name are empty.
Raises error if only one of project_name and run_name is passed in.


* **Parameters**

    
    * **project_name** (`Optional`[`str`]) – The project name. Default to current project if not passed in.


    * **run_name** (`Optional`[`str`]) – The run name. Default to current run if not passed in.



* **Return type**

    `Dict`[`str`, `Any`]



* **Returns**

    Dict[str, Any]. Response will have key status with value
    corresponding to the status of the latest job for the run.
    Other info, such as created_at, may be included.



### init(task_type, project_name=None, run_name=None, is_public=True, overwrite_local=True)
Start a run

Initialize a new run and new project, initialize a new run in an existing project,
or reinitialize an existing run in an existing project.

Before creating the project, check:
- The user is valid, login if not
- The DQ client version is compatible with API version

Optionally provide project and run names to create a new project/run or restart
existing ones.


* **Return type**

    `None`



* **Parameters**

    **task_type** (`str`) – The task type for modeling. This must be one of the valid


dataquality.schemas.task_type.TaskType options
:type project_name: `Optional`[`str`]
:param project_name: The project name. If not passed in, a random one will be
generated. If provided, and the project does not exist, it will be created. If it
does exist, it will be set.
:type run_name: `Optional`[`str`]
:param run_name: The run name. If not passed in, a random one will be
generated. If provided, and the project does not exist, it will be created. If it
does exist, it will be set.
:type is_public: `bool`
:param is_public: Boolean value that sets the project’s visibility. Default True.
:type overwrite_local: `bool`
:param overwrite_local: If True, the current project/run log directory will be
cleared during this function. If logging over many sessions with checkpoints, you
may want to set this to False. Default True


### log_data_sample(\*, text, id, \*\*kwargs)
Log a single input example to disk

Fields are expected singular elements. Field names are in the singular of
log_input_samples (texts -> text)
The expected arguments come from the task_type being used: See dq.docs() for details


* **Parameters**

    
    * **text** (`str`) – List[str] the input samples to your model


    * **id** (`int`) – List[int | str] the ids per sample


    * **split** – Optional[str] the split for this data. Can also be set via
    dq.set_split


    * **kwargs** (`Any`) – See dq.docs() for details on other task specific parameters



* **Return type**

    `None`



### log_data_samples(\*, texts, ids, meta=None, \*\*kwargs)
Logs a batch of input samples for model training/test/validation/inference.

Fields are expected as lists of their content. Field names are in the plural of
log_input_sample (text -> texts)
The expected arguments come from the task_type being used: See dq.docs() for details

ex (text classification):
.. code-block:: python

> all_labels = [“A”, “B”, “C”]
> dq.set_labels_for_run(labels = all_labels)

> texts: List[str] = [

>     “Text sample 1”,
>     “Text sample 2”,
>     “Text sample 3”,
>     “Text sample 4”

> ]

> labels: List[str] = [“B”, “C”, “A”, “A”]

> meta = {

>     “sample_importance”: [“high”, “low”, “low”, “medium”]
>     “quality_ranking”: [9.7, 2.4, 5.5, 1.2]

> }

> ids: List[int] = [0, 1, 2, 3]
> split = “training”

> dq.log_data_samples(texts=texts, labels=labels, ids=ids, meta=meta split=split)


* **Parameters**

    
    * **texts** (`List`[`str`]) – List[str] the input samples to your model


    * **ids** (`List`[`int`]) – List[int | str] the ids per sample


    * **split** – Optional[str] the split for this data. Can also be set via
    dq.set_split


    * **meta** (`Optional`[`Dict`[`str`, `List`[`Union`[`str`, `float`, `int`]]]]) – Dict[str, List[str | int | float]]. Log additional metadata fields to



* **Return type**

    `None`


each sample. The name of the field is the key of the dictionary, and the values are
a list that correspond in length and order to the text samples.
:type kwargs: `Any`
:param kwargs: See dq.docs() for details on other task specific parameters


### log_dataset(dataset, \*, batch_size=100000, text='text', id='id', split=None, meta=None, \*\*kwargs)
Log an iterable or other dataset to disk. Useful for logging memory mapped files

Dataset provided must be an iterable that can be traversed row by row, and for each
row, the fields can be indexed into either via string keys or int indexes. Pandas
and Vaex dataframes are also allowed, as well as HuggingFace Datasets

valid examples:

    d = [

        {“my_text”: “sample1”, “my_labels”: “A”, “my_id”: 1, “sample_quality”: 5.3},
        {“my_text”: “sample2”, “my_labels”: “A”, “my_id”: 2, “sample_quality”: 9.1},
        {“my_text”: “sample3”, “my_labels”: “B”, “my_id”: 3, “sample_quality”: 2.7},

    ]
    dq.log_dataset(

    > d, text=”my_text”, id=”my_id”, label=”my_labels”, meta=[“sample_quality”]

    )

    Logging a pandas dataframe, df:

        text label  id  sample_quality

    0  sample1     A   1             5.3
    1  sample2     A   2             9.1
    2  sample3     B   3             2.7
    # We don’t need to set text id or label because it matches the default
    dq.log_dataset(d, meta=[“sample_quality”])

    Logging and iterable of tuples:
    d = [

    > (“sample1”, “A”, “ID1”),
    > (“sample2”, “A”, “ID2”),
    > (“sample3”, “B”, “ID3”),

    ]
    dq.log_dataset(d, text=0, id=2, label=1)

Invalid example:

    d = {

        “my_text”: [“sample1”, “sample2”, “sample3”],
        “my_labels”: [“A”, “A”, “B”],
        “my_id”: [1, 2, 3],
        “sample_quality”: [5.3, 9.1, 2.7]

    }

In the invalid case, use dq.log_data_samples:

    meta = {“sample_quality”: d[“sample_quality”]}
    dq.log_data_samples(

    > texts=d[“my_text”], labels=d[“my_labels”], ids=d[“my_ids”], meta=meta

    )

Keyword arguments are specific to the task type. See dq.docs() for details


* **Return type**

    `None`



* **Parameters**

    **dataset** (`TypeVar`(`DataSet`, bound= `Union`[`Iterable`, `DataFrame`, `DataFrame`])) – The iterable or dataframe to log



* **Batch_size**

    The number of data samples to log at a time. Useful when logging a


memory mapped dataset. A larger batch_size will result in faster logging at the
expense of more memory usage. Default 100,000
:type text: `Union`[`str`, `int`]
:param text: str | int The column, key, or int index for text data. Default “text”
:type id: `Union`[`str`, `int`]
:param id: str | int The column, key, or int index for id data. Default “id”
:type split: `Optional`[`Split`]
:param split: Optional[str] the split for this data. Can also be set via

> dq.set_split


* **Parameters**

    
    * **meta** (`Optional`[`List`[`Union`[`str`, `int`]]]) – List[str | int] Additional keys/columns to your input data to be
    logged as metadata. Consider a pandas dataframe, this would be the list of
    columns corresponding to each metadata field to log


    * **kwargs** (`Any`) – See help(dq.get_data_logger().log_dataset) for more details here


or dq.docs() for more general task details


### log_model_outputs(\*, embs, ids, split=None, epoch=None, logits=None, probs=None, inference_name=None, exclude_embs=False)
Logs model outputs for model during training/test/validation.


* **Parameters**

    
    * **embs** (`Union`[`List`, `ndarray`, `None`]) – The embeddings per output sample


    * **ids** (`Union`[`List`, `ndarray`]) – The ids for each sample. Must match input ids of logged samples


    * **split** (`Optional`[`Split`]) – The current split. Must be set either here or via dq.set_split


    * **epoch** (`Optional`[`int`]) – The current epoch. Must be set either here or via dq.set_epoch


    * **logits** (`Union`[`List`, `ndarray`, `None`]) – The logits for each sample


    * **probs** (`Union`[`List`, `ndarray`, `None`]) – Deprecated, use logits. If passed in, a softmax will NOT be applied


    * **inference_name** (`Optional`[`str`]) – Inference name indicator for this inference split.
    If logging for an inference split, this is required.


    * **exclude_embs** (`bool`) – Optional flag to exclude embeddings from logging. If True and
    embs is set to None, this will generate random embs for each sample.



* **Return type**

    `None`


The expected argument shapes come from the task_type being used
See dq.docs() for more task specific details on parameter shape


### login()
Log into your Galileo environment.

The function will prompt your for an Authorization Token (api key) that you can
access from the console.

To skip the prompt for automated workflows, you can set GALILEO_USERNAME
(your email) and GALILEO_PASSWORD if you signed up with an email and password


* **Return type**

    `None`



### register_run_report(conditions, emails)
Register conditions and emails for a run report.

After a run is finished, a report will be sent to the specified emails.


* **Return type**

    `None`



### set_console_url(console_url=None)
For Enterprise users. Set the console URL to your Galileo Environment.

You can also set GALILEO_CONSOLE_URL before importing dataquality to bypass this


* **Return type**

    `None`



* **Parameters**

    **console_url** (`Optional`[`str`]) – If set, that will be used. Otherwise, if an environment variable


GALILEO_CONSOLE_URL is set, that will be used. Otherwise, you will be prompted for
a url.


### set_epoch(epoch)
Set the current epoch.

When set, logging model outputs will use this if not logged explicitly


* **Return type**

    `None`



### set_epoch_and_split(epoch, split, inference_name=None)
Set the current epoch and set the current split.
When set, logging data inputs/model outputs will use this if not logged explicitly
When setting split to inference, inference_name must be included


* **Return type**

    `None`



### set_labels_for_run(labels)
Creates the mapping of the labels for the model to their respective indexes.


* **Return type**

    `None`



* **Parameters**

    **labels** (`Union`[`List`[`List`[`str`]], `List`[`str`]]) – An ordered list of labels (ie [‘dog’,’cat’,’fish’]


If this is a multi-label type, then labels are a list of lists where each inner
list indicates the label for the given task

This order MUST match the order of probabilities that the model outputs.

In the multi-label case, the outer order (order of the tasks) must match the
task-order of the task-probabilities logged as well.


### set_split(split, inference_name=None)
Set the current split.

When set, logging data inputs/model outputs will use this if not logged explicitly
When setting split to inference, inference_name must be included


* **Return type**

    `None`



### set_tagging_schema(tagging_schema)
Sets the tagging schema for NER models

Only valid for text_ner task_types. Others will throw an exception


* **Return type**

    `None`



### set_tasks_for_run(tasks, binary=True)
Sets the task names for the run (multi-label case only).

This order MUST match the order of the labels list provided in log_input_data
and the order of the probability vectors provided in log_model_outputs.

This also must match the order of the labels logged in set_labels_for_run (meaning
that the first list of labels must be the labels of the first task passed in here)


* **Parameters**

    
    * **tasks** (`List`[`str`]) – The list of tasks for your run


    * **binary** (`bool`) – Whether this is a binary multi label run. If true, tasks will also



* **Return type**

    `None`


be set as your labels, and you should NOT call dq.set_labels_for_run it will be
handled for you. Default True


### wait_for_run(project_name=None, run_name=None)
Waits until a specific project run transitions from started to finished.
Defaults to the current run if project_name and run_name are empty.
Raises error if only one of project_name and run_name is passed in.


* **Parameters**

    
    * **project_name** (`Optional`[`str`]) – The project name. Default to current project if not passed in.


    * **run_name** (`Optional`[`str`]) – The run name. Default to current run if not passed in.



* **Return type**

    `None`



* **Returns**

    None. Function returns after the run transitions to finished


# Indices and tables


* [Index](genindex.md)


* [Module Index](py-modindex.md)


* [Search Page](search.md)
