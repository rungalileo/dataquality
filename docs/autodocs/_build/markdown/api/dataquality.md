# dataquality

dataquality


### login()
Log into your Galileo environment.

The function will prompt your for an Authorization Token (api key) that you can
access from the console.

To skip the prompt for automated workflows, you can set GALILEO_USERNAME
(your email) and GALILEO_PASSWORD if you signed up with an email and password


* **Return type**

    `None`



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

    **task_type** (`str`) -- The task type for modeling. This must be one of the valid


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
:param is_public: Boolean value that sets the project's visibility. Default True.
:type overwrite_local: `bool`
:param overwrite_local: If True, the current project/run log directory will be
cleared during this function. If logging over many sessions with checkpoints, you
may want to set this to False. Default True


### log_model_outputs(\*, embs, ids, split=None, epoch=None, logits=None, probs=None, inference_name=None, exclude_embs=False)
Logs model outputs for model during training/test/validation.


* **Parameters**

    
    * **embs** (`Union`[`List`, `ndarray`, `None`]) -- The embeddings per output sample


    * **ids** (`Union`[`List`, `ndarray`]) -- The ids for each sample. Must match input ids of logged samples


    * **split** (`Optional`[`Split`]) -- The current split. Must be set either here or via dq.set_split


    * **epoch** (`Optional`[`int`]) -- The current epoch. Must be set either here or via dq.set_epoch


    * **logits** (`Union`[`List`, `ndarray`, `None`]) -- The logits for each sample


    * **probs** (`Union`[`List`, `ndarray`, `None`]) -- Deprecated, use logits. If passed in, a softmax will NOT be applied


    * **inference_name** (`Optional`[`str`]) -- Inference name indicator for this inference split.
    If logging for an inference split, this is required.


    * **exclude_embs** (`bool`) -- Optional flag to exclude embeddings from logging. If True and
    embs is set to None, this will generate random embs for each sample.



* **Return type**

    `None`


The expected argument shapes come from the task_type being used
See dq.docs() for more task specific details on parameter shape


### finish(last_epoch=None, wait=True, create_data_embs=False)
Finishes the current run and invokes a job


* **Parameters**

    
    * **last_epoch** (`Optional`[`int`]) -- If set, only epochs up to this value will be uploaded/processed
    This is inclusive, so setting last_epoch to 5 would upload epochs 0,1,2,3,4,5


    * **wait** (`bool`) -- If true, after uploading the data, this will wait for the
    run to be processed by the Galileo server. If false, you can manually wait
    for the run by calling dq.wait_for_run() Default True


    * **create_data_embs** (`bool`) -- If True, an off-the-shelf transformer will run on the raw
    text input to generate data-level embeddings. These will be available in the
    data view tab of the Galileo console. You can also access these embeddings
    via dq.metrics.get_data_embeddings()



* **Return type**

    `str`



### set_labels_for_run(labels)
Creates the mapping of the labels for the model to their respective indexes.
:rtype: `None`


* **Parameters**

    **labels** (`Union`[`List`[`List`[`str`]], `List`[`str`]]) -- An ordered list of labels (ie ['dog','cat','fish']


If this is a multi-label type, then labels are a list of lists where each inner
list indicates the label for the given task

This order MUST match the order of probabilities that the model outputs.

In the multi-label case, the outer order (order of the tasks) must match the
task-order of the task-probabilities logged as well.


### set_tasks_for_run(tasks, binary=True)
Sets the task names for the run (multi-label case only).

This order MUST match the order of the labels list provided in log_input_data
and the order of the probability vectors provided in log_model_outputs.

This also must match the order of the labels logged in set_labels_for_run (meaning
that the first list of labels must be the labels of the first task passed in here)


* **Return type**

    `None`



* **Parameters**

    
    * **tasks** (`List`[`str`]) -- The list of tasks for your run


    * **binary** (`bool`) -- Whether this is a binary multi label run. If true, tasks will also


be set as your labels, and you should NOT call dq.set_labels_for_run it will be
handled for you. Default True


### set_epoch(epoch)
Set the current epoch.

When set, logging model outputs will use this if not logged explicitly


* **Return type**

    `None`



### set_split(split, inference_name=None)
Set the current split.

When set, logging data inputs/model outputs will use this if not logged explicitly
When setting split to inference, inference_name must be included


* **Return type**

    `None`



### log_data_sample(\*, text, id, \*\*kwargs)
Log a single input example to disk

Fields are expected singular elements. Field names are in the singular of
log_input_samples (texts -> text)
The expected arguments come from the task_type being used: See dq.docs() for details


* **Parameters**

    
    * **text** (`str`) -- List[str] the input samples to your model


    * **id** (`int`) -- List[int | str] the ids per sample


    * **split** -- Optional[str] the split for this data. Can also be set via
    dq.set_split


    * **kwargs** (`Any`) -- See dq.docs() for details on other task specific parameters



* **Return type**

    `None`



### log_dataset(dataset, \*, batch_size=100000, text='text', id='id', split=None, meta=None, \*\*kwargs)
Log an iterable or other dataset to disk. Useful for logging memory mapped files

Dataset provided must be an iterable that can be traversed row by row, and for each
row, the fields can be indexed into either via string keys or int indexes. Pandas
and Vaex dataframes are also allowed, as well as HuggingFace Datasets

valid examples:

    d = [

        {"my_text": "sample1", "my_labels": "A", "my_id": 1, "sample_quality": 5.3},
        {"my_text": "sample2", "my_labels": "A", "my_id": 2, "sample_quality": 9.1},
        {"my_text": "sample3", "my_labels": "B", "my_id": 3, "sample_quality": 2.7},

    ]
    dq.log_dataset(

    > d, text="my_text", id="my_id", label="my_labels", meta=["sample_quality"]

    )

    Logging a pandas dataframe, df:

        text label  id  sample_quality

    0  sample1     A   1             5.3
    1  sample2     A   2             9.1
    2  sample3     B   3             2.7
    # We don't need to set text id or label because it matches the default
    dq.log_dataset(d, meta=["sample_quality"])

    Logging and iterable of tuples:
    d = [

    > ("sample1", "A", "ID1"),
    > ("sample2", "A", "ID2"),
    > ("sample3", "B", "ID3"),

    ]
    dq.log_dataset(d, text=0, id=2, label=1)

Invalid example:

    d = {

        "my_text": ["sample1", "sample2", "sample3"],
        "my_labels": ["A", "A", "B"],
        "my_id": [1, 2, 3],
        "sample_quality": [5.3, 9.1, 2.7]

    }

In the invalid case, use dq.log_data_samples:

    meta = {"sample_quality": d["sample_quality"]}
    dq.log_data_samples(

    > texts=d["my_text"], labels=d["my_labels"], ids=d["my_ids"], meta=meta

    )

Keyword arguments are specific to the task type. See dq.docs() for details


* **Parameters**

    
    * **dataset** (`TypeVar`(`DataSet`, bound= `Union`[`Iterable`, `DataFrame`, `DataFrame`])) -- The iterable or dataframe to log


    * **text** (`Union`[`str`, `int`]) -- str | int The column, key, or int index for text data. Default "text"


    * **id** (`Union`[`str`, `int`]) -- str | int The column, key, or int index for id data. Default "id"


    * **split** (`Optional`[`Split`]) -- Optional[str] the split for this data. Can also be set via
    dq.set_split


    * **meta** (`Optional`[`List`[`Union`[`str`, `int`]]]) -- List[str | int] Additional keys/columns to your input data to be
    logged as metadata. Consider a pandas dataframe, this would be the list of


    * **kwargs** (`Any`) -- See help(dq.get_data_logger().log_dataset) for more details here



* **Batch_size**

    The number of data samples to log at a time. Useful when logging a
    memory mapped dataset. A larger batch_size will result in faster logging at the
    expense of more memory usage. Default 100,000



* **Return type**

    `None`
    columns corresponding to each metadata field to log


or dq.docs() for more general task details


### auto(hf_data=None, hf_inference_names=None, train_data=None, val_data=None, test_data=None, inference_data=None, max_padding_length=200, hf_model='distilbert-base-uncased', labels=None, project_name=None, run_name=None, wait=True, create_data_embs=False)
Automatically gets insights on a text classification or NER dataset

Given either a pandas dataframe, file_path, or huggingface dataset path, this
function will load the data, train a huggingface transformer model, and
provide Galileo insights via a link to the Galileo Console

One of hf_data, train_data should be provided. If neither of those are, a
demo dataset will be loaded by Galileo for training.


* **Parameters**

    
    * **hf_data** (`Union`[`DatasetDict`, `str`, `None`]) -- Union[DatasetDict, str] Use this param if you have huggingface
    data in the hub or in memory. Otherwise see train_data, val_data,
    and test_data. If provided, train_data, val_data, and test_data are ignored.


    * **hf_inference_names** (`Optional`[`List`[`str`]]) -- Use this param alongside hf_data if you have splits
    you'd like to consider as inference. A list of key names in hf_data
    to be run as inference runs after training. Any keys set must exist in hf_data


    * **train_data** (`Union`[`DataFrame`, `Dataset`, `str`, `None`]) -- Optional training data to use. Can be one of
    \* Pandas dataframe
    \* Huggingface dataset
    \* Path to a local file
    \* Huggingface dataset hub path


    * **val_data** (`Union`[`DataFrame`, `Dataset`, `str`, `None`]) -- Optional validation data to use. The validation data is what is
    used for the evaluation dataset in huggingface, and what is used for early
    stopping. If not provided, but test_data is, that will be used as the evaluation
    set. If neither val nor test are available, the train data will be randomly
    split 80/20 for use as evaluation data.
    Can be one of
    \* Pandas dataframe
    \* Huggingface dataset
    \* Path to a local file
    \* Huggingface dataset hub path


    * **test_data** (`Union`[`DataFrame`, `Dataset`, `str`, `None`]) -- Optional test data to use. The test data, if provided with val,
    will be used after training is complete, as the held-out set. If no validation
    data is provided, this will instead be used as the evaluation set.
    Can be one of
    \* Pandas dataframe
    \* Huggingface dataset
    \* Path to a local file
    \* Huggingface dataset hub path


    * **inference_data** (`Optional`[`Dict`[`str`, `Union`[`DataFrame`, `Dataset`, `str`]]]) -- User this param to include inference data alongside the
    train_data param. If you are passing data via the hf_data parameter, you
    should use the hf_inference_names param. Optional inference datasets to run
    with after training completes. The structure is a dictionary with the
    key being the inference name and the value one of
    \* Pandas dataframe
    \* Huggingface dataset
    \* Path to a local file
    \* Huggingface dataset hub path


    * **max_padding_length** (`int`) -- The max length for padding the input text
    during tokenization. Default 200


    * **hf_model** (`str`) -- The pretrained AutoModel from huggingface that will be used to
    tokenize and train on the provided data. Default distilbert-base-uncased


    * **labels** (`Optional`[`List`[`str`]]) -- Optional list of labels for this dataset. If not provided, they
    will attempt to be extracted from the data


    * **project_name** (`Optional`[`str`]) -- Optional project name. If not set, a random name will
    be generated


    * **run_name** (`Optional`[`str`]) -- Optional run name for this data. If not set, a random name will
    be generated


    * **wait** (`bool`) -- Whether to wait for Galileo to complete processing your run.
    Default True


    * **create_data_embs** (`bool`) -- Whether to create data embeddings for this run. If True,
    Sentence-Transformers will be used to generate data embeddings for this dataset
    and uploaded with this run. You can access these embeddings via
    dq.metrics.get_data_embeddings in the emb column or
    dq.metrics.get_dataframe(..., include_data_embs=True) in the data_emb col
    Only available for TC currently. NER coming soon. Default False.



* **Return type**

    `None`


For text classification datasets, the only required columns are text and label

For NER, the required format is the huggingface standard format of tokens and
tags (or ner_tags).
See example: [https://huggingface.co/datasets/rungalileo/mit_movies](https://huggingface.co/datasets/rungalileo/mit_movies)

> MIT Movies dataset in huggingface format

```python
tokens                                              ner_tags
[what, is, a, good, action, movie, that, is, r...       [0, 0, 0, 0, 7, 0, ...
[show, me, political, drama, movies, with, jef...       [0, 0, 7, 8, 0, 0, ...
[what, are, some, good, 1980, s, g, rated, mys...       [0, 0, 0, 0, 5, 6, ...
[list, a, crime, film, which, director, was, d...       [0, 0, 7, 0, 0, 0, ...
[is, there, a, thriller, movie, starring, al, ...       [0, 0, 0, 7, 0, 0, ...
...                                               ...                      ...
```

To see auto insights on a random, pre-selected dataset, simply run

```python
import dataquality as dq

dq.auto()
```

An example using auto with a hosted huggingface text classification dataset

```python
import dataquality as dq

dq.auto(hf_data="rungalileo/trec6")
```

Similarly, for NER

```python
import dataquality as dq

dq.auto(hf_data="conll2003")
```

An example using auto with sklearn data as pandas dataframes

```python
import dataquality as dq
import pandas as pd
from sklearn.datasets import fetch_20newsgroups

# Load the newsgroups dataset from sklearn
newsgroups_train = fetch_20newsgroups(subset='train')
newsgroups_test = fetch_20newsgroups(subset='test')
# Convert to pandas dataframes
df_train = pd.DataFrame(
    {"text": newsgroups_train.data, "label": newsgroups_train.target}
)
df_test = pd.DataFrame(
    {"text": newsgroups_test.data, "label": newsgroups_test.target}
)

dq.auto(
     train_data=df_train,
     test_data=df_test,
     labels=newsgroups_train.target_names,
     project_name="newsgroups_work",
     run_name="run_1_raw_data"
)
```

An example of using auto with a local CSV file with text and label columns

```python
import dataquality as dq

dq.auto(
    train_data="train.csv",
    test_data="test.csv",
    project_name="data_from_local",
    run_name="run_1_raw_data"
)
```

# dataquality.integrations.torch


### watch(model, dataloaders=[], classifier_layer=None, embedding_dim=None, logits_dim=None, embedding_fn=None, logits_fn=None, last_hidden_state_layer=None, unpatch_on_start=False)
wraps a PyTorch model and optionally dataloaders to log the
embeddings and logits to [Galileo]([https://www.rungalileo.io/](https://www.rungalileo.io/)).

```python
dq.log_dataset(train_dataset, split="train")
train_dataloader = torch.utils.data.DataLoader()
model = TextClassificationModel(num_labels=len(train_dataset.list_of_labels))
watch(model, [train_dataloader, test_dataloader])
for epoch in range(NUM_EPOCHS):
    dq.set_epoch_and_split(epoch,"training")
    train()
    dq.set_split("validation")
    validate()
dq.finish()
```


* **Parameters**

    
    * **model** (`Module`) -- Pytorch Model to be wrapped


    * **dataloaders** (`Optional`[`List`[`DataLoader`]]) -- List of dataloaders to be wrapped


    * **classifier_layer** (`Union`[`Module`, `str`, `None`]) -- Layer to hook into (usually 'classifier' or 'fc').
    Inputs are the embeddings and outputs are the logits.


    * **embedding_dim** (`Union`[`str`, `int`, `slice`, `Tensor`, `List`, `Tuple`, `None`]) -- Dimension of the embeddings for example "[:, 0]"
    to remove the cls token


    * **logits_dim** (`Union`[`str`, `int`, `slice`, `Tensor`, `List`, `Tuple`, `None`]) -- Dimension to extract the logits for example in NER
    "[:,1:,:]"


    * **logits_dim** -- Dimension of the logits
    from layer input and logits from layer output. If the layer is not found,
    the last_hidden_state_layer will be used


    * **embedding_fn** (`Optional`[`Callable`]) -- Function to process embeddings from the model


    * **logits_fn** (`Optional`[`Callable`]) -- Function to process logits from the model f.e.
    lambda x: x[0]


    * **last_hidden_state_layer** (`Union`[`Module`, `str`, `None`]) -- Layer to extract the embeddings from


    * **unpatch_on_start** (`bool`) -- Force unpatching of dataloaders
    instead of global patching


    * **model** -- Pytorch Model to be wrapped


    * **dataloaders** -- List of dataloaders to be wrapped


    * **last_hidden_state_layer** -- Layer to extract the embeddings from


    * **embedding_dim** -- Dimension of the embeddings for example "[:, 0]"



* **Return type**

    `None`


to remove the cls token
:param logits_dim: Dimension to extract the logits for example in NER

> "[:,1:,:]"


### unwatch(model=None, force=True)
Unwatches the model. Run after the run is finished.
:type force: `bool`
:param force: Force unwatch even if the model is not watched


* **Return type**

    `None`


# dataquality.integrations.transformers_trainer


### watch(trainer, last_hidden_state_layer=None, embedding_dim=None, logits_dim=None, classifier_layer=None, embedding_fn=None, logits_fn=None)
used to *hook* into to the **trainer**
to log to [Galileo]([https://www.rungalileo.io/](https://www.rungalileo.io/))


* **Parameters**

    **trainer** (`Trainer`) -- Trainer object



* **Return type**

    `None`



### unwatch(trainer)
unwatch is used to remove the callback from the trainer
:type trainer: `Trainer`
:param trainer: Trainer object


* **Return type**

    `None`


# dataquality.integrations.keras


### _class_ DataQualityCallback()
Bases: `Callback`


#### on_epoch_begin(epoch, logs)
Called at the start of an epoch.

Subclasses should override for any actions to run. This function should only
be called during TRAIN mode.


* **Parameters**

    
    * **epoch** (`int`) -- Integer, index of epoch.


    * **logs** (`Dict`) -- Dict. Currently no data is passed to this argument for this method
    but that may change in the future.



* **Return type**

    `None`



#### on_test_batch_begin(batch, logs=None)
Called at the beginning of a batch in evaluate methods.

Also called at the beginning of a validation batch in the fit
methods, if validation data is provided.

Subclasses should override for any actions to run.

Note that if the steps_per_execution argument to compile in
tf.keras.Model is set to N, this method will only be called every N
batches.


* **Parameters**

    
    * **batch** (`Any`) -- Integer, index of batch within the current epoch.


    * **logs** (`Optional`[`Dict`]) -- Dict. Currently no data is passed to this argument for this method
    but that may change in the future.



* **Return type**

    `None`



#### on_test_batch_end(batch, logs=None)
Called at the end of a batch in evaluate methods.

Also called at the end of a validation batch in the fit
methods, if validation data is provided.

Subclasses should override for any actions to run.

Note that if the steps_per_execution argument to compile in
tf.keras.Model is set to N, this method will only be called every N
batches.


* **Parameters**

    
    * **batch** (`Any`) -- Integer, index of batch within the current epoch.


    * **logs** (`Optional`[`Dict`]) -- Dict. Aggregated metric results up until this batch.



* **Return type**

    `None`



#### on_train_batch_begin(batch, logs=None)
Called at the beginning of a training batch in fit methods.

Subclasses should override for any actions to run.

Note that if the steps_per_execution argument to compile in
tf.keras.Model is set to N, this method will only be called every N
batches.


* **Parameters**

    
    * **batch** (`Any`) -- Integer, index of batch within the current epoch.


    * **logs** (`Optional`[`Dict`]) -- Dict. Currently no data is passed to this argument for this method
    but that may change in the future.



* **Return type**

    `None`



#### on_train_batch_end(batch, logs=None)
Called at the end of a training batch in fit methods.

Subclasses should override for any actions to run.

Note that if the steps_per_execution argument to compile in
tf.keras.Model is set to N, this method will only be called every N
batches.


* **Parameters**

    
    * **batch** (`Any`) -- Integer, index of batch within the current epoch.


    * **logs** (`Optional`[`Dict`]) -- Dict. Aggregated metric results up until this batch.



* **Return type**

    `None`



### _class_ DataQualityLoggingLayer(what_to_log)
Bases: `Layer`


#### call(inputs)
This is where the layer's logic lives.

The call() method may not create state (except in its first invocation,
wrapping the creation of variables or other resources in tf.init_scope()).
It is recommended to create state in __init__(), or the build() method
that is called automatically before call() executes the first time.


* **Parameters**

    
    * **inputs** (`Tensor`) -- Input tensor, or dict/list/tuple of input tensors.
    The first positional inputs argument is subject to special rules:
    - inputs must be explicitly passed. A layer cannot have zero

    > arguments, and inputs cannot be provided via the default value
    > of a keyword argument.


        * NumPy array or Python scalar values in inputs get cast as tensors.


        * Keras mask metadata is only collected from inputs.


        * Layers are built (build(input_shape) method)
    using shape info from inputs only.


        * input_spec compatibility is only checked against inputs.


        * Mixed precision input casting is only applied to inputs.
    If a layer has tensor arguments in \*args or \*\*kwargs, their
    casting behavior in mixed precision should be handled manually.


        * The SavedModel input specification is generated using inputs only.


        * Integration with various ecosystem packages like TFMOT, TFLite,
    TF.js, etc is only supported for inputs and not for tensors in
    positional and keyword arguments.



    * **\*args** -- Additional positional arguments. May contain tensors, although
    this is not recommended, for the reasons above.


    * **\*\*kwargs** -- Additional keyword arguments. May contain tensors, although
    this is not recommended, for the reasons above.
    The following optional keyword arguments are reserved:
    - training: Boolean scalar tensor of Python boolean indicating

    > whether the call is meant for training or inference.


        * mask: Boolean input mask. If the layer's call() method takes a
    mask argument, its default value will be set to the mask generated
    for inputs by the previous layer (if input did come from a layer
    that generated a corresponding mask, i.e. if it came from a Keras
    layer with masking support).




* **Return type**

    `Tensor`



* **Returns**

    A tensor or list/tuple of tensors.



### add_ids_to_numpy_arr(orig_arr, ids)
Deprecated, use add_sample_ids


* **Return type**

    `ndarray`



### add_sample_ids(orig_arr, ids)
Add sample IDs to the training/test data before training begins

This is necessary to call before training a Keras model with the
Galileo DataQualityCallback


* **Return type**

    `ndarray`



* **Parameters**

    
    * **orig_arr** (`ndarray`) -- The numpy array to be passed into model.train


    * **ids** (`Union`[`List`[`int`], `ndarray`]) -- The ids for each sample to append. These are the same IDs that are


logged for the input data. They must match 1-1

# dataquality.integrations.experimental.keras


### watch(model, layer=None, seed=42)
Watch a model and log the inputs and outputs of a layer.
:type model: `Layer`
:param model: The model to watch
:type layer: `Optional`[`Any`]
:param layer: The layer to watch, if None the classifier layer is used
:type seed: `int`
:param seed: The seed to use for the model


* **Return type**

    `None`



### unwatch(model)
Unpatches the model. Run after the run is finished
:type model: `Layer`
:param model: The model to unpatch


* **Return type**

    `None`


# dataquality.integrations.spacy


### watch(nlp)
Stores the nlp object before calling watch on the ner component within it

We need access to the nlp object so that during training we can capture the
model's predictions over the raw text by running nlp("user's text") and looking
at the results


* **Parameters**

    **nlp** (`Language`) -- The spacy nlp Language component.



* **Return type**

    `None`



### unwatch(nlp)
Returns spacy nlp Language component to its original unpatched state.

Unfortunately, spacy does not make this easy, so we replicate spacy's add_pipe
for logic for using internal spacy methods to add a component object to a specific
position.


* **Return type**

    `None`


# dataquality.integrations.hf


### infer_schema(label_list)
Infers the schema via the exhaustive list of labels


* **Return type**

    `TaggingSchema`



### tokenize_and_log_dataset(dd, tokenizer, label_names=None, meta=None)
This function tokenizes a huggingface DatasetDict and aligns the labels to BPE

After tokenization, this function will also log the dataset(s) present in the
DatasetDict


* **Parameters**

    
    * **dd** (`DatasetDict`) -- DatasetDict from huggingface to log


    * **tokenizer** (`PreTrainedTokenizerBase`) -- The pretrained tokenizer from huggingface


    * **label_names** (`Optional`[`List`[`str`]]) -- Optional list of labels for the dataset. These can typically
    be extracted automatically (if the dataset came from hf datasets hub or was
    exported via Galileo dataquality). If they cannot be extracted, an error will
    be raised requesting label names


    * **meta** (`Optional`[`List`[`str`]]) -- Optional metadata columns to be logged. The columns must be present
    in at least one of the splits of the dataset.



* **Return type**

    `DatasetDict`



### _class_ TextDataset(hf_dataset)
Bases: `Dataset`

An abstracted Huggingface Text dataset for users to import and use

Get back a DataLoader via the get_dataloader function


### get_dataloader(dataset, \*\*kwargs)
Create a DataLoader for a particular split given a huggingface Dataset

The DataLoader will be a loader of a TextDataset. The __getitem__ for that dataset
will return:

> 
> * id - the Galileo ID of the sample


> * input_ids - the standard huggingface input_ids


> * attention_mask - the standard huggingface attention_mask


> * labels - output labels adjusted with tokenized NER data


* **Parameters**

    
    * **dataset** (`Dataset`) -- The huggingface dataset to convert to a DataLoader


    * **kwargs** (`Any`) -- Any additional keyword arguments to be passed into the DataLoader
    Things like batch_size or shuffle



* **Return type**

    `DataLoader`


# dataquality

dataquality


### _class_ AggregateFunction(value)
Bases: `str`, `Enum`

An enumeration.


### _class_ Operator(value)
Bases: `str`, `Enum`

An enumeration.


### _class_ Condition(\*\*data)
Bases: `BaseModel`

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
If the aggregate function is "pct", you don't need to specify a metric,

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
...     agg=AggregateFunction.avg,
...     metric="confidence",
...     operator=Operator.lt,
...     threshold=0.3,
... )
>>> c(df)  # Will raise an AssertionError if False


* **Parameters**

    
    * **metric** -- The DF column for evaluating the condition


    * **agg** -- An aggregate function to apply to the metric


    * **operator** -- The operator to use for comparing the agg to the threshold
    (e.g. "gt", "lt", "eq", "neq")


    * **threshold** -- Threshold value for evaluating the condition


    * **filter** -- Optional filter to apply to the DataFrame before evaluating the
    condition



### _class_ ConditionFilter(\*\*data)
Bases: `BaseModel`

Filter a dataframe based on the column value

Note that the column used for filtering is the same as the metric used
in the condition.


* **Parameters**

    
    * **operator** -- The operator to use for filtering (e.g. "gt", "lt", "eq", "neq")
    See Operator


    * **value** -- The value to compare against
