from typing import Any, Dict, List

from torch.nn import Module
from transformers.data.data_collator import DataCollatorWithPadding
from transformers.trainer_callback import TrainerCallback  # noqa: E402
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments  # noqa: E402

import dataquality as dq
from dataquality.exceptions import GalileoException
from dataquality.schemas.split import Split
from dataquality.utils.hf_datasets import load_pandas_df

from datasets import Dataset 
# Trainer callback for Huggingface transformers Trainer library
class DQCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that sends the logs to [Galileo](https://www.rungalileo.io/).
    """

    def __init__(self) -> None:
        self.helper_data = dq.get_model_logger().logger_config.helper_data
        self._initialized = False

    def _clear_logger_config_helper_data(self) -> None:
        self.helper_data["embs"] = None
        self.helper_data["probs"] = None
        self.helper_data["logits"] = None

    def setup(
        self, args: TrainingArguments, state: TrainerState, model: Module, kwargs: Any
    ) -> None:
        if args.remove_unused_columns is None or args.remove_unused_columns:
            raise GalileoException(
                "TrainerArgument remove_unused_columns must be false"
            )
        self._dq = dq
        self._attach_hooks_to_model(model)
        eval_dataloader = kwargs["eval_dataloader"]
        train_dataloader = kwargs["train_dataloader"]
        self.tokenizer = kwargs["tokenizer"]

        # 🔭🌕 Galileo logging

        dq.log_dataset(train_dataloader.dataset,split=Split.train) #id=train_dataloader.dataset["idx"]
        # convert to pandas not needed
        if getattr(eval_dataloader, "dataset", False):
            dq.log_dataset(
                load_pandas_df(eval_dataloader.dataset), split=Split.validation
            )
        dq.set_labels_for_run(train_dataloader.dataset.features["label"].names)
        self._initialized = True

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        if not self._initialized:
            self.setup(args, state, kwargs["model"], kwargs)
        self._dq.set_split(Split.train)  # 🔭🌕 Galileo logging

    def on_epoch_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        state_epoch = state.epoch
        if state_epoch is not None:
            state_epoch = int(state_epoch)
            self._dq.set_epoch(state_epoch)  # 🔭🌕 Galileo logging

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        self._dq.set_split(Split.validation)  # 🔭🌕 Galileo logging

    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        pass

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        """
        Perform a training step on a batch of inputs.
        """
        # Log only if embedding exists

        assert(self.helper_data.get("embs",None) is not None,"Embedding passed to the logger can not be logged")
        assert(self.helper_data.get("logits",None) is not None,"Logits passed to the logger can not be logged")
        assert(self.helper_data.get("idx",None) is not None,"idx column missing in dataset (needed to map rows to the indices/ids)")

        # 🔭🌕 Galileo logging
        self._dq.log_model_outputs(**self.helper_data)

    def _attach_hooks_to_model(self, model: Module) -> None:
        next(model.children()).register_forward_hook(self._embedding_hook)
        model.register_forward_hook(self._logit_hook)

    def _embedding_hook(
        self, model: Module, model_input: Any, model_output: Any
    ) -> None:
        #TODO EMBEDDING CHECK
        if hasattr(model_input,"last_hidden_state"):
            output_detached = model_input.last_hidden_state.detach()
        else:
            output_detached = model_output.last_hidden_state.detach()


        self.helper_data["embs"] = output_detached[:, 0]

    def _logit_hook(self, model: Module, model_input: Any, model_output: Any) -> None:
        # log the output logits
        self.helper_data["logits"] = model_output.logits.detach()

    def add_idx_col(self, dataset:Dataset) -> None:
       dataset  = dataset.add_column("idx",list(range(len(dataset))))
       dataset  = dataset.add_column("id",list(range(len(dataset))))
       return dataset

    # workaround to save the idx
    # TODO: find cleaner way
    # ADD arguments
    def collate_fn(self, rows: List[Dict]) -> Dict[str, Any]:
        # in: ['text', 'label', 'idx', 'input_ids', 'attention_mask']
        indices = [row.get("idx") for row in rows]
        self.helper_data["ids"] = indices
        cleaned_rows = [
            {
                "label": row.get("label"),
                "input_ids": row.get("input_ids"),
                "attention_mask": row.get("attention_mask"),
            }
            for row in rows
        ]

        # out: ['label', 'input_ids', 'attention_mask']
        return DataCollatorWithPadding(self.tokenizer)(cleaned_rows)
