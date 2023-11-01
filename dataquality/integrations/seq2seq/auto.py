from random import choice
from typing import List, Optional, Tuple

from datasets import Dataset, DatasetDict, load_dataset
from transformers import PreTrainedModel

import dataquality as dq
from dataquality.dq_auto.base_data_manager import BaseDatasetManager
from dataquality.integrations.seq2seq.formatters import get_formatter
from dataquality.integrations.seq2seq.s2s_trainer import do_train, get_trainer
from dataquality.integrations.seq2seq.schema import (
    Seq2SeqDatasetConfig,
    Seq2SeqGenerationConfig,
    Seq2SeqTrainingConfig,
)
from dataquality.schemas.split import Split
from dataquality.schemas.task_type import TaskType
from dataquality.utils.auto import (
    add_val_data_if_missing,
    get_meta_cols,
    run_name_from_hf_dataset,
    sample_dataset_dict,
)
from dataquality.utils.torch import cleanup_cuda


class S2SDatasetManager(BaseDatasetManager):
    DEMO_DATASETS = [
        "tatsu-lab/alpaca",
        # "billsum",  # TODO: add billsum
    ]

    def try_load_dataset_dict_from_config(
        self,
        dataset_config: Optional[Seq2SeqDatasetConfig],
    ) -> Tuple[Optional[DatasetDict], Seq2SeqDatasetConfig]:
        """Tries to load the DatasetDict if available

        If the user provided the hf_data param we load it from huggingface
        If they provided nothing, we load the demo dataset
        Otherwise, we return None, because the user provided train/test/val data, and
        that requires task specific processing

        For HF datasets, we optionally apply a formatting function to the dataset to
        convert it to the format we expect. This is useful for datasets that have
        non-standard columns, like the `alpaca` dataset, which has `instruction`,
        `input`, and `target` columns instead of `text` and `label`
        """
        if not dataset_config:
            hf_data = choice(self.DEMO_DATASETS)
            print(f"No dataset provided, using {hf_data} for run")
            dataset_config = Seq2SeqDatasetConfig(hf_data=hf_data)

        if dataset_config.hf_data:
            hf_data = dataset_config.hf_data
            if isinstance(hf_data, str):
                dd = load_dataset(hf_data)
                dataset_config.formatter = get_formatter(hf_data)
            elif isinstance(hf_data, DatasetDict):
                dd = hf_data
            else:
                raise ValueError(
                    "hf_data must be a path to a huggingface DatasetDict in the hf "
                    "hub or a DatasetDict object. "
                    "If this is just a Dataset, pass it to `train_data`"
                )

            dataset_config.input_col = dataset_config.formatter.input_col
            dataset_config.target_col = dataset_config.formatter.target_col
            return dd, dataset_config

        return None, dataset_config

    def get_dataset_dict_from_config(
        self,
        dataset_config: Optional[Seq2SeqDatasetConfig],
        max_train_size: Optional[int] = None,
        create_val_data_if_missing: bool = True,
    ) -> Tuple[DatasetDict, Seq2SeqDatasetConfig]:
        """Creates and/or validates the DatasetDict provided by the user.

        If a user provides a DatasetDict, we simply validate it. Otherwise, we
        parse a combination of the parameters provided, generate a DatasetDict of their
        training data, and validate that.

        If the user provides hf_data, we load that dataset from huggingface and
        optionally apply a formatting function to the dataset to convert it to the
        format we expect. This is useful for datasets that have non-standard columns,
        like the `alpaca` dataset, which has `instruction`, `input`, and `target`
        columns instead of `text` and `label`

        If the user provides train_path, val_path, or test_path, we load those files
        and convert them to a DatasetDict.

        Else if the user provides train_data, val_data, or test_data, we convert those
        to a DatasetDict.
        """
        dd, dataset_config = self.try_load_dataset_dict_from_config(dataset_config)
        dd = dd or DatasetDict()

        if not dd:
            train_data = dataset_config.train_path or dataset_config.train_data
            # We don't need to check for train data in dd because
            # `try_load_dataset_dict_from_config` validates that it exists already
            dd[Split.train] = self._convert_to_hf_dataset(train_data)

            val_data = dataset_config.val_path or dataset_config.val_data
            if val_data is not None:
                dd[Split.validation] = self._convert_to_hf_dataset(val_data)

            test_data = dataset_config.test_path or dataset_config.test_data
            if test_data is not None:
                dd[Split.test] = self._convert_to_hf_dataset(test_data)

        # Minimize dataset if user provided a max_train_size
        dd = sample_dataset_dict(dd, dataset_config, max_train_size)
        # Add validation data if missing, add 'id' column
        dd = self._validate_dataset_dict(
            dd, [], create_val_data_if_missing=create_val_data_if_missing
        )
        formatter = dataset_config.formatter
        if formatter.process_batch:
            # Apply the dataset's custom formatter on dataset dict
            dd = dd.map(
                formatter.format_batch,
                batched=True,
                remove_columns=dd[Split.train].column_names,
                with_indices=True,
            )
            # We must re-add the id column if it's been dropped
            dd = self._validate_dataset_dict(
                dd, [], create_val_data_if_missing=create_val_data_if_missing
            )
        else:
            dd = dd.map(formatter.format_sample, remove_columns=formatter.remove_cols)

        return dd, dataset_config

    def _validate_dataset_dict(
        self,
        dd: DatasetDict,
        inference_names: List[str],
        labels: Optional[List[str]] = None,
        create_val_data_if_missing: bool = True,
    ) -> DatasetDict:
        """Validates the core components of the provided (or created) DatasetDict

        The DatasetDict that the user provides or that we create from the provided
        train/test/val data must have the following:
            * all keys must be one of our valid key names

        We then also convert the keys of the DatasetDict to our `Split` key enum so
        we can access it easier in the future.

        If create_val_data_if_missing=True, we additionally ensure that we have
        validation data. If validation data is missing, we use the test split as
        validation (if available), or else we create a validation split from the
        training data.
        """
        clean_dd = super()._validate_dataset_dict(dd, inference_names, labels)
        for key in list(clean_dd.keys()):
            ds = clean_dd.pop(key)
            if "id" not in ds.features:
                ds = ds.add_column("id", list(range(ds.num_rows)))
            clean_dd[key] = ds

        if create_val_data_if_missing:
            clean_dd = add_val_data_if_missing(clean_dd, TaskType.seq2seq)

        return clean_dd


def _log_dataset_dict(dd: DatasetDict, input_col: str, target_col: str) -> None:
    for key in dd.keys():
        ds: Dataset = dd[key]
        if key in Split.get_valid_keys():
            if input_col != "text" and "text" in ds.column_names:
                ds = ds.rename_columns({"text": "_metadata_text"})
            if target_col != "label" and "label" in ds.column_names:
                ds = ds.rename_columns({"label": "_metadata_label"})
            if input_col != "input" and "input" in ds.column_names:
                ds = ds.rename_columns({"input": "_metadata_input"})
            if target_col != "target" and "target" in ds.column_names:
                ds = ds.rename_columns({"target": "_metadata_target"})
            meta = get_meta_cols(ds.features, {input_col, target_col})
            dq.log_dataset(ds, text=input_col, label=target_col, split=key, meta=meta)


def auto(
    project_name: str = "auto_s2s",
    run_name: Optional[str] = None,
    dataset_config: Optional[Seq2SeqDatasetConfig] = None,
    training_config: Optional[Seq2SeqTrainingConfig] = None,
    generation_config: Optional[Seq2SeqGenerationConfig] = None,
    max_train_size: Optional[int] = None,
    wait: bool = True,
) -> Optional[PreTrainedModel]:
    """Automatically get insights on a Seq2Seq dataset

    Given either a pandas dataframe, file_path, or huggingface dataset path, this
    function will load the data, train a huggingface transformer model, and
    provide Galileo insights via a link to the Galileo Console. If the number of epochs
    in training_config is set to 0, training/fine-tuning will be skipped and we will
    only do a forward pass (on all the splits).

    One of DatasetConfig `hf_data`, `train_path`, or `train_data` should be provided.
    If none of those is, a demo dataset will be loaded by Galileo for training.

    The validation data is what is used for the evaluation dataset in training.
    If not provided, but test_data is, that will be used as the evaluation
    set. If neither val nor test are available (and training is not skipped), the train
    data will be randomly split 80/20 for use as evaluation data.

    The test data, if provided with val, will be used after training is complete, as the
    hold-out set. If no validation data is provided, this will instead be used as the
    evaluation set.

    :param project_name: Optional project name. If not set, a random name will
        be generated
    :param run_name: Optional run name for this data. If not set, a random name will
        be generated
    :param dataset_config: Optional config for loading the dataset.
        See `Seq2SeqDatasetConfig` for more details
    :param training_config: Optional config for training the model.
        See `Seq2SeqTrainingConfig` for more details
    :param generation_config: Optional config for generating predictions.
        See `Seq2SeqGenerationConfig` for more details
    :param max_train_size: Optional max number of training examples to use.
    :param wait: Whether to wait for Galileo to complete processing your run.
        Default True

    To see auto insights on a random, pre-selected dataset, simply run
    ```python
        from dataquality.integrations.seq2seq import auto

        auto()
    ```

    An example using `auto` with a hosted huggingface dataset
    ```python
        from dataquality.integrations.seq2seq.auto import auto
        from dataquality.integrations.seq2seq.schema import Seq2SeqDatasetConfig

        dataset_config = Seq2SeqDatasetConfig(hf_data="tatsu-lab/alpaca")
        auto(dataset_config=dataset_config)
    ```

    An example of using `auto` with a local file with `text` and `label` columns
    ```python
    from dataquality.integrations.seq2seq.auto import auto
    from dataquality.integrations.seq2seq.schema import AutoDatasetConfig

    dataset_config = Seq2SeqDatasetConfig(
        train_path="train.jsonl", eval_path="eval.jsonl"
    )
    auto(
        project_name="s2s_auto",
        run_name="completion_dataset"
        dataset_config=dataset_config,
    )
    ```
    """

    training_config = training_config or Seq2SeqTrainingConfig()
    generation_config = generation_config or Seq2SeqGenerationConfig()

    manager = S2SDatasetManager()
    # Only force the creation of validation data if we actually fine-tune the model
    create_val_data_if_missing = training_config.epochs > 0
    dd, dataset_config = manager.get_dataset_dict_from_config(
        dataset_config, max_train_size, create_val_data_if_missing
    )

    if not run_name and isinstance(dataset_config.hf_data, str):
        run_name = run_name_from_hf_dataset(dataset_config.hf_data)

    dq.login()
    dq.init(TaskType.seq2seq, project_name=project_name, run_name=run_name)
    input_col = dataset_config.input_col
    target_col = dataset_config.target_col
    # We 'watch' in get_trainer, which must happen before logging datasets
    model, dataloaders = get_trainer(
        dd,
        dataset_config.input_col,
        dataset_config.target_col,
        training_config,
        generation_config,
    )

    _log_dataset_dict(dd, input_col=input_col, target_col=target_col)
    model = do_train(model, dataloaders, training_config, wait)
    if training_config.return_model:
        return model

    # Cleanup and return None if we don't want to return the model
    cleanup_cuda(model=model)
    return None
