# Imports for the hook manager
from typing import Any, Callable, Dict, Optional
from warnings import warn

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
from dataquality.schemas.torch import DimensionSlice, HelperData, InputDim, Layer
from dataquality.utils.helpers import check_noop
from dataquality.utils.patcher import Cleanup, Patch, PatchManager, RefManager
from dataquality.utils.torch import ModelHookManager, remove_all_forward_hooks
from dataquality.utils.transformers import RemoveIdCollatePatch, SignatureColumnsPatch

a = Analytics(ApiClient, dq.config)  # type: ignore
a.log_import("transformers_trainer")


class DQCallback(TrainerCallback, TorchBaseInstance, Patch):
    """
    [`TrainerCallback`] that provides data quality insights
    with [Galileo](https://www.rungalileo.io/). This callback
    is logs during each training training step and is using the Huggingface
    transformers Trainer library.
    """

    hook_manager: ModelHookManager

    def __init__(
        self,
        trainer: Trainer,
        last_hidden_state_layer: Optional[Layer] = None,
        embedding_dim: Optional[InputDim] = None,
        logits_dim: Optional[InputDim] = None,
        classifier_layer: Layer = "classifier",
        embedding_fn: Optional[Callable] = None,
        logits_fn: Optional[Callable] = None,
        helper_data: Dict[str, Any] = {},
    ) -> None:
        # Access the dq logger helper data
        self.helper_data = helper_data
        self.helper_data[HelperData.model_outputs_store] = {}
        self.model_outputs_store = self.helper_data[HelperData.model_outputs_store]
        self._initialized = False
        self._model_setup = False
        # Hook manager for attaching hooks to the model
        self.hook_manager = ModelHookManager()
        self.last_hidden_state_layer = last_hidden_state_layer
        self.classifier_layer = classifier_layer
        self.embedding_fn = embedding_fn
        self.logits_fn = logits_fn
        self._init_dimension(embedding_dim, logits_dim)
        self.trainer = trainer

    def _clear_logger_config_curr_model_outputs(self) -> None:
        self.model_outputs_store.clear()

    def _do_log(self) -> None:
        # Log only if embedding exists
        assert self.model_outputs_store.get("embs") is not None, GalileoException(
            "Embedding passed to the logger can not be logged"
        )
        assert self.model_outputs_store.get("logits") is not None, GalileoException(
            "Logits passed to the logger can not be logged"
        )
        assert self.model_outputs_store.get("ids") is not None, GalileoException(
            "Did you map IDs to your dataset before watching the model? You can run:\n"
            "`ds= dataset.map(lambda x, idx: {'id': idx}, with_indices=True)`\n"
            "id (index) column is needed in the dataset for logging"
        )

        # 🔭🌕 Galileo logging
        dq.log_model_outputs(**self.model_outputs_store)
        self._clear_logger_config_curr_model_outputs()

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
        self._clear_logger_config_curr_model_outputs()

    def validate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        """Validate the model and dataset"""
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
            raise GalileoException(
                f"Unknown dataset type {type(train_dataloader_ds)}. "
                "Must be a datasets.Dataset or torch.utils.data.Dataset. "
                "Each row must be dictionary with the id columns. "
                'For example: return {"input_ids": ..., "attention_mask":'
                ' ..., "id": ...}'
            )
        self._initialized = True

    def setup_model(self, model: Module) -> None:
        """Setup the callback
        :param args: Training arguments
        :param state: Trainer state
        :param model: Model
        :param kwargs: Keyword arguments
            (eval_dataloader, train_dataloader, tokenizer)"""
        if self._model_setup:
            return
        assert dq.config.task_type, GalileoException(
            "dq client must be initialized. "
            "For example: dq.init('text_classification')"
        )
        self.task = dq.config.task_type
        # Attach hooks to the model
        self._attach_hooks_to_model(
            model, self.classifier_layer, self.last_hidden_state_layer
        )
        self._model_setup = True

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
        """
        if not self._initialized:
            self.validate(args, state, control, **kwargs)
            self._initialized = True
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
        """
        self._do_log()

    def _attach_hooks_to_model(
        self, model: Module, classifier_layer: Layer, last_hidden_state_layer: Layer
    ) -> None:
        """
        Method to attach hooks to the model by using the hook manager
        :param model: Model
        :param model: pytorch model layer to attach hooks to
        """
        try:
            self.hook_manager.attach_classifier_hook(
                model, self._classifier_hook, classifier_layer
            )
        except Exception as e:
            warn(
                "Could not attach function to model layer. Error:"
                f" {e}. Please check that the classifier layer name:"
                f" {classifier_layer} exists in the model. Common layers"
                " to extract logits and the last hidden state are 'classifier'"
                "and 'fc'. To fix this, pass the correct layer name to the "
                "'classifier_layer' parameter in the 'watch' function. "
                "For example: 'watch(model, classifier_layer='fc')'."
                "You can view the model layers by using the 'model.named_children'"
                "function or by printing the model."
            )
            self.hook_manager.attach_hooks_to_model(
                model, self._dq_embedding_hook, last_hidden_state_layer
            )
            self.hook_manager.attach_hook(model, self._dq_logit_hook)

    def _patch(self) -> "Patch":
        """Patch the trainer to add the callback"""
        self.trainer.add_callback(self)
        return self

    def _unpatch(self) -> None:
        """Unpatch the trainer to remove the callback"""
        self.trainer.remove_callback(self)


@check_noop
def watch(
    trainer: Trainer,
    last_hidden_state_layer: Optional[Layer] = None,
    embedding_dim: Optional[DimensionSlice] = None,
    logits_dim: Optional[DimensionSlice] = None,
    classifier_layer: Optional[Layer] = None,
    embedding_fn: Optional[Callable] = None,
    logits_fn: Optional[Callable] = None,
) -> None:
    """used to *hook* into to the **trainer**
    to log to [Galileo](https://www.rungalileo.io/)

    :param trainer: Trainer object
    """
    a.log_function("transformers_trainer/watch")
    helper_data = dq.get_model_logger().logger_config.helper_data
    # Callback which we add to the trainer
    dqcallback = DQCallback(
        trainer=trainer,
        last_hidden_state_layer=last_hidden_state_layer,
        embedding_dim=embedding_dim,
        logits_dim=logits_dim,
        classifier_layer=classifier_layer,
        embedding_fn=embedding_fn,
        logits_fn=logits_fn,
        helper_data=helper_data,
    )
    # The columns needed for the forward process
    signature_patch = SignatureColumnsPatch(trainer)
    signature_cols = signature_patch.new_signature_columns
    dqcallback.patch()
    assert trainer.args.n_gpu <= 1, GalileoException(
        "Parallel GPUs are not supported. TrainingArguments.n_gpu should be set to 1"
    )
    # We wrap the data collator to remove the id column
    RemoveIdCollatePatch(
        trainer,
        signature_cols,
        dqcallback.helper_data[HelperData.model_outputs_store],
    )
    # Save the original signature columns and the callback for unwatch
    dqcallback.setup_model(trainer.model)
    # Unpatch Trainer after logging (when finished is called)
    cleanup_manager = RefManager(lambda: unwatch(trainer))
    helper_data["cleaner"] = Cleanup(cleanup_manager)


def unwatch(trainer: Trainer) -> None:
    """
    `unwatch` is used to remove the callback from the trainer
    :param trainer: Trainer object
    """
    a.log_function("transformers_trainer/unwatch")
    # helper_data = dq.get_model_logger().logger_config.helper_data
    manager = PatchManager()
    manager.unpatch()
    remove_all_forward_hooks(trainer.model)
