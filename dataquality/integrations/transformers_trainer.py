# Imports for the hook manager
from typing import Callable, Dict, Optional

from datasets import Dataset
from torch.nn import Module
from torch.utils.data import Dataset as TorchDataset
from transformers import Trainer
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

import dataquality as dq
from dataquality.analytics import Analytics
from dataquality.clients.api import ApiClient
from dataquality.exceptions import GalileoException
from dataquality.integrations.torch import TorchBaseInstance
from dataquality.schemas.split import Split
from dataquality.schemas.torch import DimensionSlice, InputDim, Layer
from dataquality.utils.helpers import check_noop
from dataquality.utils.torch import ModelHookManager
from dataquality.utils.transformers import (
    add_id_to_signature_columns,
    remove_id_collate_fn_wrapper,
)

a = Analytics(ApiClient, dq.config)  # type: ignore
a.log_import("transformers_trainer")


class DQCallback(TrainerCallback, TorchBaseInstance):
    """
    [`TrainerCallback`] that provides data quality insights
    with [Galileo](https://www.rungalileo.io/). This callback
    is logs during each training training step and is using the Huggingface
    transformers Trainer library.
    """

    hook_manager: ModelHookManager

    def __init__(
        self,
        last_hidden_state_layer: Layer = None,
        embedding_dim: InputDim = None,
        logits_dim: InputDim = None,
        classifier_layer: Layer = "classifier",
        embedding_fn: Optional[Callable] = None,
        logits_fn: Optional[Callable] = None,
    ) -> None:
        # Access the dq logger helper data
        self.helper_data = dq.get_model_logger().logger_config.helper_data
        self.helper_data["model_outputs"] = {}
        self.model_outputs = self.helper_data["model_outputs"]
        self._initialized = False
        # Hook manager for attaching hooks to the model
        self.hook_manager = ModelHookManager()
        self.last_hidden_state_layer = last_hidden_state_layer
        self.classifier_layer = classifier_layer
        self.embedding_fn = embedding_fn
        self.logits_fn = logits_fn
        self._init_dimension(embedding_dim, logits_dim)

    def _clear_logger_config_helper_data(self) -> None:
        self.model_outputs.clear()

    def _do_log(self) -> None:
        # Log only if embedding exists
        assert self.model_outputs.get("embs") is not None, GalileoException(
            "Embedding passed to the logger can not be logged"
        )
        assert self.model_outputs.get("logits") is not None, GalileoException(
            "Logits passed to the logger can not be logged"
        )
        assert self.model_outputs.get("ids") is not None, GalileoException(
            "Did you map IDs to your dataset before watching the model? You can run:\n"
            "`ds= dataset.map(lambda x, idx: {'id': idx}, with_indices=True)`\n"
            "id (index) column is needed in the dataset for logging"
        )

        # 🔭🌕 Galileo logging
        dq.log_model_outputs(**self.model_outputs)
        self._clear_logger_config_helper_data()

    def on_init_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Dict,
    ) -> None:
        """
        Event called at the end of the initialization of the [`Trainer`].
        """
        self._clear_logger_config_helper_data()

    def setup(
        self,
        args: TrainingArguments,
        state: TrainerState,
        kwargs: Dict,
    ) -> None:
        """Setup the callback
        :param args: Training arguments
        :param state: Trainer state
        :param model: Model
        :param kwargs: Keyword arguments (eval_dataloader, train_dataloader, tokenizer)
        :return: None"""

        assert dq.config.task_type, GalileoException(
            "dq client must be initialized. "
            "For example: dq.init('text_classification')"
        )
        self.task = dq.config.task_type
        model: Module = kwargs["model"]
        # Attach hooks to the model
        self._attach_hooks_to_model(
            model, self.classifier_layer, self.last_hidden_state_layer
        )
        train_dataloader = kwargs["train_dataloader"]
        train_dataloader_ds = train_dataloader.dataset
        if isinstance(train_dataloader_ds, Dataset):
            assert "id" in train_dataloader_ds.column_names, GalileoException(
                "Did you map IDs to your dataset before watching the model?\n"
                "To add the id column with datasets. You can run:\n"
                """`ds= dataset.map(lambda x, idx: {"id": idx},"""
                " with_indices=True)`. The id (index) column is needed in "
                "the dataset for logging"
            )
        elif isinstance(train_dataloader_ds, TorchDataset):
            item = next(iter(train_dataloader_ds))
            assert hasattr(item, "keys") and "id" in item.keys(), GalileoException(
                "Dataset __getitem__ needs to return a dictionary"
                " including the index id. "
                'For example: return {"input_ids": ..., "attention_mask":'
                ' ..., "id": ...}'
            )
        else:
            raise GalileoException(f"Unknown dataset type {type(train_dataloader_ds)}")
        self._initialized = True

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Dict,
    ) -> None:
        """
        Event called at the beginning of training. Attaches hooks to model.
        :param args: Training arguments
        :param state: Trainer state
        :param control: Trainer control
        :param kwargs: Keyword arguments (model, eval_dataloader, tokenizer...)
        :return: None
        """
        if not self._initialized:
            self.setup(args, state, kwargs)
        dq.set_split(Split.training)  # 🔭🌕 Galileo logging

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Dict,
    ) -> None:
        dq.set_split(Split.validation)

    def on_epoch_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Dict,
    ) -> None:
        state_epoch = state.epoch
        if state_epoch is not None:
            state_epoch = int(state_epoch)
            dq.set_epoch(state_epoch)  # 🔭🌕 Galileo logging
        dq.set_split(Split.training)

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Dict,
    ) -> None:
        dq.set_split(Split.validation)

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Dict,
    ) -> None:
        dq.set_split(Split.test)

    def on_prediction_step(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Dict,
    ) -> None:
        self._do_log()

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Dict,
    ) -> None:
        """
        Perform a training step on a batch of inputs.
        Log the embeddings, ids and logits.
        :param args: Training arguments
        :param state: Trainer state
        :param control: Trainer control
        :param kwargs: Keyword arguments (including the model, inputs, outputs)
        :return: None
        """
        self._do_log()

    def _attach_hooks_to_model(
        self, model: Module, classifier_layer: Layer, last_hidden_state_layer: Layer
    ) -> None:
        """
        Method to attach hooks to the model by using the hook manager
        :param model: Model
        :param model: pytorch model layer to attach hooks to
        :return: None
        """
        try:
            self.hook_manager.attach_classifier_hook(
                model, self._classifier_hook, classifier_layer
            )
        except Exception as e:
            print(
                f"Could not attach classifier hook to model. "
                f"Error: {e}. "
                f"Please check the classifier layer name: {classifier_layer}"
            )
            self.hook_manager.attach_hooks_to_model(
                model, self._embedding_hook, last_hidden_state_layer
            )
            self.hook_manager.attach_hook(model, self._logit_hook)


@check_noop
def watch(
    trainer: Trainer,
    last_hidden_state_layer: Layer = None,
    embedding_dim: Optional[DimensionSlice] = None,
    logits_dim: Optional[DimensionSlice] = None,
    classifier_layer: Layer = "classifier",
    embedding_fn: Optional[Callable] = None,
    logits_fn: Optional[Callable] = None,
) -> None:
    """
    [`watch`] is used to hook into to the trainer
    to log to [Galileo](https://www.rungalileo.io/)
    :param trainer: Trainer object
    :return: None
    """
    a.log_function("transformers_trainer/watch")

    # Callback which we add to the trainer
    dqcallback = DQCallback(
        last_hidden_state_layer=last_hidden_state_layer,
        embedding_dim=embedding_dim,
        logits_dim=logits_dim,
        classifier_layer=classifier_layer,
        embedding_fn=embedding_fn,
        logits_fn=logits_fn,
    )
    # The columns needed for the forward process
    signature_cols = add_id_to_signature_columns(trainer)

    assert trainer.args.n_gpu <= 1, GalileoException(
        "Parallel GPUs are not supported. TrainingArguments.n_gpu should be set to 1"
    )
    # We wrap the data collator to remove the id column
    trainer.data_collator = remove_id_collate_fn_wrapper(
        trainer.data_collator, signature_cols, dqcallback.helper_data["model_outputs"]
    )
    trainer.add_callback(dqcallback)
