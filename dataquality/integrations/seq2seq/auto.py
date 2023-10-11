from typing import List, Optional

from datasets import Dataset, DatasetDict
from transformers import Trainer

import dataquality as dq
from dataquality.dq_auto.base_data_manager import BaseDatasetManager
from dataquality.integrations.seq2seq.s2s_trainer import do_train, get_trainer
from dataquality.integrations.seq2seq.schema import (
    AutoDatasetConfig,
    AutoGenerationConfig,
    AutoTrainingConfig,
)
from dataquality.schemas.split import Split
from dataquality.schemas.task_type import TaskType
from dataquality.utils.auto import (
    add_val_data_if_missing,
    run_name_from_hf_dataset,
)


class S2SDatasetManager(BaseDatasetManager):
    DEMO_DATASETS = [
        "tatsu-lab/alpaca",
        # "billsum",  # TODO: add billsum
    ]

    def _validate_dataset_dict(
        self,
        dd: DatasetDict,
        inference_names: List[str],
        labels: Optional[List[str]] = None,
    ) -> DatasetDict:
        """Validates the core components of the provided (or created) DatasetDict

        The DatasetDict that the user provides or that we create from the provided
        train/test/val data must have the following:
            * all keys must be one of our valid key names

        We then also convert the keys of the DatasetDict to our `Split` key enum so
        we can access it easier in the future
        """
        clean_dd = super()._validate_dataset_dict(dd, inference_names, labels)
        for key in list(clean_dd.keys()):
            ds = clean_dd.pop(key)
            # TODO: temporary, update
            ds = ds.select(range(100))

            if "id" not in ds.features:
                ds = ds.add_column("id", list(range(ds.num_rows)))
            clean_dd[key] = ds
        return add_val_data_if_missing(clean_dd, TaskType.seq2seq)


def _log_dataset_dict(dd: DatasetDict, input_col: str, target_col: str) -> None:
    for key in dd.keys():
        ds: Dataset = dd[key]
        if key in Split.get_valid_keys():
            if input_col != "text" and "text" in ds.column_names:
                ds = ds.rename_columns({"text": "_metadata_text"})
            if target_col != "label" and "label" in ds.column_names:
                ds = ds.rename_columns({"label": "_metadata_label"})

            dq.log_dataset(ds, text=input_col, label=target_col, split=key)


def auto(
    project_name: str = "auto_s2s",
    run_name: Optional[str] = None,
    dataset_config: Optional[AutoDatasetConfig] = None,
    training_config: Optional[AutoTrainingConfig] = None,
    generation_config: Optional[AutoGenerationConfig] = None,
    wait: bool = True,
) -> Trainer:
    """Automatically get insights on a Seq2Seq dataset

    Given either a pandas dataframe, file_path, or huggingface dataset path, this
    function will load the data, train a huggingface transformer model, and
    provide Galileo insights via a link to the Galileo Console

    One of DatasetConfig `hf_data`, `train_path`, or `train_data` should be provided.
    If none of those is, a demo dataset will be loaded by Galileo for training.

    The validation data is what is used for the evaluation dataset in huggingface.
    If not provided, but test_data is, that will be used as the evaluation
    set. If neither val nor test are available, the train data will be randomly
    split 80/20 for use as evaluation data.

    The test data, if provided with val,
    will be used after training is complete, as the held-out set. If no validation
    data is provided, this will instead be used as the evaluation set.

    :param project_name: Optional project name. If not set, a random name will
        be generated
    :param run_name: Optional run name for this data. If not set, a random name will
        be generated
    :param dataset_config: Optional config for loading the dataset.
        See `AutoDatasetConfig` for more details
    :param training_config: Optional config for training the model.
        See `AutoTrainingConfig` for more details
    :param generation_config: Optional config for generating predictions.
        See `AutoGenerationConfig` for more details
    :param wait: Whether to wait for Galileo to complete processing your run.
        Default True

    To see auto insights on a random, pre-selected dataset, simply run
    ```python
        from dataquality.integrations.seq2seq import auto

        auto()
    ```

    An example using `auto` with a hosted huggingface dataset
    ```python
        from dataquality.integrations.seq2seq.schema import AutoDatasetConfig
        from dataquality.integrations.seq2seq import auto

        dataset_config = AutoDatasetConfig(hf_data="tatsu-lab/alpaca")
        auto(dataset_config=dataset_config)
    ```

    An example of using `auto` with a local file with `text` and `label` columns
    ```python
    from dataquality.integrations.seq2seq.schema import AutoDatasetConfig
    from dataquality.integrations.seq2seq import auto

    dataset_config = AutoDatasetConfig(
        train_path="train.jsonl", eval_path="eval.jsonl"
    )
    auto(
        project_name="data_from_local",
        run_name="run_1_raw_data"
        dataset_config=dataset_config,
    )
    ```
    """
    dataset_config = dataset_config or AutoDatasetConfig()
    training_config = training_config or AutoTrainingConfig()
    generation_config = generation_config or AutoGenerationConfig()

    manager = S2SDatasetManager()
    dd = manager.get_dataset_dict(
        dataset_config.hf_data,
        train_data=dataset_config.train_data,
        val_data=dataset_config.val_data,
        test_data=dataset_config.test_data,
    )
    dq.login()
    if not run_name and isinstance(dataset_config.hf_data, str):
        run_name = run_name_from_hf_dataset(dataset_config.hf_data)

    dq.init(TaskType.seq2seq, project_name=project_name, run_name=run_name)
    input_col = manager.formatter.input_col
    target_col = manager.formatter.target_col

    # We 'watch' in get_trainer, which must happen before logging datasets
    model, dataloaders = get_trainer(
        dd,
        dataset_config.input_col,
        dataset_config.target_col,
        training_config,
        generation_config,
    )

    _log_dataset_dict(dd, input_col=input_col, target_col=target_col)
    return do_train(model, dataloaders, training_config, wait)
