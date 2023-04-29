from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import pandas as pd
from datasets import ClassLabel, Dataset, DatasetDict

import dataquality as dq
from dataquality import Analytics, ApiClient
from dataquality.dq_auto.base_data_manager import BaseDatasetManager
from dataquality.dq_auto.ic_trainer import get_trainer
from dataquality.exceptions import GalileoException
from dataquality.schemas.split import Split
from dataquality.schemas.task_type import TaskType
from dataquality.utils.auto import add_val_data_if_missing, run_name_from_hf_dataset, _find_col_name
from dataquality.utils.auto_trainer import do_train

a = Analytics(ApiClient, dq.config)
a.log_import("auto_ic")



class ICDatasetManager(BaseDatasetManager):
    DEMO_DATASETS = ["cifar10", "mnist", "beans"]
    DEFAULT_CHECKPOINT = "google/vit-base-patch16-224"

    def _add_class_label_to_dataset(
        self, ds: Dataset, labels: Optional[List[str]] = None
    ) -> Dataset:
        """Map a not ClassLabel 'label' column to a ClassLabel, if possible"""
        if "label" not in ds.features or isinstance(ds.features["label"], ClassLabel):
            return ds
        labels = labels if labels is not None else sorted(set(ds["label"]))
        # For string columns, map the label2idx so we can cast to ClassLabel
        if ds.features["label"].dtype == "string":
            label_to_idx = dict(zip(labels, range(len(labels))))
            ds = ds.map(lambda row: {"label": label_to_idx[row["label"]]})

        # https://github.com/python/mypy/issues/6239
        class_label = ClassLabel(num_classes=len(labels), names=labels)  # type: ignore
        ds = ds.cast_column("label", class_label)
        return ds

    def _validate_dataset_dict(
        self,
        dd: DatasetDict,
        inference_names: List[str],
        labels: Optional[List[str]] = None,
    ) -> DatasetDict:
        """Validates the core components of the provided (or created) DatasetDict)

        The DatasetDict that the user provides or that we create from the provided
        train/test/val data must have the following:
            * all keys must be one of our valid key names
            * it must have either
                * a `image`, `img` or `images` column containing the image
                * a column containing the word `path` containing a (relative) path
            * it must have a `label` or `labels` column
                * if this column isn't a ClassLabel, we convert it to one

        We then also convert the keys of the DatasetDict to our `Split` key enum so
        we can access it easier in the future
        """
        # TODO: rename the cols to be standard (img -> image)
        # return dd
        label_colnames = set()

        clean_dd = super()._validate_dataset_dict(dd, inference_names, labels)
        for key in list(clean_dd.keys()):
            ds = clean_dd.pop(key)
            
            # Find image column and rename it to "image" or "path"
            try:
                imgs_colname = _find_col_name(["img", "image", "images"], ds.column_names)
                if imgs_colname != "image":
                    ds = ds.rename_column(imgs_colname, "image")
            except GalileoException:
                imgs_location_colname = _find_col_name(["path"], ds.column_names, method="include")
                if imgs_location_colname != "path": # TODO: maybe shouldn't rename and just pass it forward ?
                    ds = ds.rename_column(imgs_location_colname, "path")

            # Find label column, make some checks and rename it to "label"
            if key not in inference_names:
                label_colname = _find_col_name(["label", "labels"], ds.column_names)
                assert label_colname is not None, f"Did not find a label column in columns {ds.column_names}"
                label_colnames.add(label_colname)
                assert len(label_colnames) == 1, f"Found multiple label columns across splits: {label_colnames}"
                if label_colname != "label": # TODO: maybe shouldn't rename and just pass it forward ?
                    ds = ds.rename_column(label_colname, "label")
                if not isinstance(ds.features["label"], ClassLabel):
                    ds = self._add_class_label_to_dataset(ds, labels)

            # Add id column if missing
            if "id" not in ds.features:
                ds = ds.add_column("id", list(range(ds.num_rows)))
            clean_dd[key] = ds
        return add_val_data_if_missing(clean_dd)

    # TODO: adapt to add loading a folder, maybe also rewrite _convert_df_to_dataset + _add_class_label_to_dataset
    def _convert_to_hf_dataset(
        self,
        data: Union[pd.DataFrame, Dataset, str],
        labels: Optional[List[str]] = None,
    ) -> Dataset:
        """Loads the data into (hf) Dataset format.

        Data can be one of Dataset, pandas df, str. If str, it's either a path to a
        file or a path to a remote huggingface Dataset that we load with `load_dataset`
        """
        if isinstance(data, Dataset):
            return data
        if isinstance(data, pd.DataFrame):
            return self._convert_df_to_dataset(data, labels)
        if isinstance(data, str):
            ds = load_data_from_str(data) # TODO redo 
            if isinstance(ds, pd.DataFrame):
                ds = self._convert_df_to_dataset(ds, labels)
            return ds
        raise GalileoException(
            "Dataset must be one of pandas DataFrame, "
            "huggingface Dataset, or string path"
        )

def _get_labels(dd: DatasetDict, labels: Optional[List[str]] = None) -> List[str]:
    """Gets the labels for this dataset from the dataset if not provided."""
    # TODO: add doc

    if isinstance(labels, (list, np.ndarray)):
        return list(labels)
    
    train_labels = dd[Split.train].features["label"]
    if hasattr(train_labels, "names"):
        return train_labels.names

    return sorted(set(dd[Split.train]["label"]))


# TODO: adapt
def _log_dataset_dict(dd: DatasetDict) -> None:
    for key in dd:
        ds = dd[key]
        default_cols = ["text", "image", "label", "label_idx", "id", "path"] # TODO: change !! clean up
        meta = [i for i in ds.features if i not in default_cols]

        imgs_colname, imgs_location_colname = None, None
        if "path" in ds.column_names: # TODO: better
            imgs_location_colname = "path"
        else:
            imgs_colname = "image"
        

        if key in Split.get_valid_keys():
            dq.log_image_dataset(dataset=ds, imgs_colname=imgs_colname, imgs_location_colname=imgs_location_colname, meta=meta, split=key)
        else:
            # TODO: test that with inference
            dq.log_image_dataset(dataset=ds, imgs_colname=imgs_colname, imgs_location_colname=imgs_location_colname, meta=meta, split=Split.inference, inference_name=key)

def auto(
    hf_data: Optional[Union[DatasetDict, str]] = None,
    hf_inference_names: Optional[List[str]] = None,
    train_data: Optional[Union[pd.DataFrame, Dataset, str]] = None,
    val_data: Optional[Union[pd.DataFrame, Dataset, str]] = None,
    test_data: Optional[Union[pd.DataFrame, Dataset, str]] = None,
    inference_data: Optional[Dict[str, Union[pd.DataFrame, Dataset, str]]] = None,
    hf_model: Optional[str] = None,
    labels: Optional[List[str]] = None,
    project_name: str = "auto_ic",
    run_name: Optional[str] = None,
    wait: bool = True,
    create_data_embs: Optional[bool] = None
) -> None:
    """TODO
    Automatically gets insights on an Image Classification dataset

    TODO
    Given either a pandas dataframe, file_path, or huggingface dataset path, this
    function will load the data, train a huggingface token classification model, and
    provide Galileo insights via a link to the Galileo Console

    One of `hf_data`, `train_data` should be provided. If neither of those are, a
    demo dataset will be loaded by Galileo for training.

    The data must be provided in the standard "huggingface" format
    * `huggingface` format: A dataset with `tokens` and (`ner_tags` or `tags`) columns
        See example: https://huggingface.co/datasets/rungalileo/mit_movies

        MIT Movies dataset in huggingface format

        tokens	                                            ner_tags
        [what, is, a, good, action, movie, that, is, r...	[0, 0, 0, 0, 7, 0, ...
        [show, me, political, drama, movies, with, jef...	[0, 0, 7, 8, 0, 0, ...
        [what, are, some, good, 1980, s, g, rated, mys...	[0, 0, 0, 0, 5, 6, ...
        [list, a, crime, film, which, director, was, d...	[0, 0, 7, 0, 0, 0, ...
        [is, there, a, thriller, movie, starring, al, ...	[0, 0, 0, 7, 0, 0, ...
        ...                                               ...                      ...

    :param hf_data: Union[DatasetDict, str] Use this param if you have huggingface
        data in the hub or in memory. Otherwise see `train_data`, `val_data`,
        and `test_data`. If provided, train_data, val_data, and test_data are ignored
    :param train_data: Optional training data to use. Can be one of
        * Pandas dataframe
        * Huggingface dataset
        * Path to a local file
        * Huggingface dataset hub path
    :param val_data: Optional validation data to use. The validation data is what is
        used for the evaluation dataset in huggingface, and what is used for early
        stopping. If not provided, but test_data is, that will be used as the evaluation
        set. If neither val nor test are available, the train data will be randomly
        split 80/20 for use as evaluation data.
        Can be one of
        * Pandas dataframe
        * Huggingface dataset
        * Path to a local file
        * Huggingface dataset hub path
    :param test_data: Optional test data to use. The test data, if provided with val,
        will be used after training is complete, as the held-out set. If no validation
        data is provided, this will instead be used as the evaluation set.
        Can be one of
        * Pandas dataframe
        * Huggingface dataset
        * Path to a local file
        * Huggingface dataset hub path
    :param hf_model: The pretrained AutoModel from huggingface that will be used to
        tokenize and train on the provided data. Default distilbert-base-uncased
    :param labels: Optional list of labels for this dataset. If not provided, they
        will attempt to be extracted from the data
    :param project_name: Optional project name. If not set, a random name will
        be generated
    :param run_name: Optional run name for this data. If not set, a random name will
        be generated
    :param wait: Whether to wait for Galileo to complete processing your run.
        Default True

    To see auto insights on a random, pre-selected dataset, simply run
    ```python
        from dataquality.auto.ner import auto

        auto()
    ```

    An example using `auto` with a hosted huggingface dataset
    ```python
        from dataquality.auto.text_classification import auto

        auto(hf_data="rungalileo/mit_movies")
    ```

    An example using `auto` with sklearn data as pandas dataframes
    ```python
        import pandas as pd
        from dataquality.auto.ner import auto

        TODO EXAMPLE FOR NER FROM PANDAS DFs

        auto(
             train_data=df_train,
             test_data=df_test,
             labels=['O','B-ACTOR','I-ACTOR','B-TITLE','I-TITLE','B-YEAR','I-YEAR']
             project_name="ner_movie_reviews",
             run_name="run_1_raw_data"
        )
    ```

    An example of using `auto` with a local CSV file with `text` and `label` columns
    ```python
    from dataquality.auto.ner import auto

    auto(
         train_data="train.csv",
         test_data="test.csv",
         project_name="data_from_local",
         run_name="run_1_raw_data"
    )
    ```
    """
    manager = ICDatasetManager()
    dd = manager.get_dataset_dict(
        hf_data,
        hf_inference_names,
        train_data,
        val_data,
        test_data,
        inference_data,
        labels,
    )

    dd[Split.train] = dd[Split.train].filter(lambda example: example["label"] in [0,1]).select(range(10))
    dd[Split.validation] = dd[Split.validation].filter(lambda example: example["label"] in [0,1]).select(range(10))
    dd[Split.test] = dd[Split.test].filter(lambda example: example["label"] in [0,1]).select(range(10))
    class_label = ClassLabel(names = list(dd[Split.train].features["label"].names)[:2])
    dd[Split.train].features["label"] = class_label
    dd[Split.validation].features["label"] = class_label
    dd[Split.test].features["label"] = class_label


    # Log in
    dq.set_console_url("https://console.dev.rungalileo.io") # TODO: remove that
    dq.login()
    a.log_function("auto/ic")

    # Find the run name and initiate the run 
    if not run_name:
        if isinstance(hf_data, str):
            run_name = run_name_from_hf_dataset(hf_data)
        elif isinstance(train_data, str):
            # TODO parse out the dir name train and try to get the datasetname : or too much ?
            run_name = run_name_from_hf_dataset(train_data)
    dq.init(TaskType.image_classification, project_name=project_name, run_name=run_name)

    # Log labels and the dataset
    labels = _get_labels(dd, labels)
    dq.set_labels_for_run(labels)
    _log_dataset_dict(dd)
    
    if hf_model is None:
        hf_model = manager.DEFAULT_CHECKPOINT
    trainer, augmented_data = get_trainer(dd, labels, hf_model)
    do_train(trainer, augmented_data, wait, create_data_embs)
