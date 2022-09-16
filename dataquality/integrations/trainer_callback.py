from transformers.trainer_callback import TrainerCallback , TrainerControl, TrainerState  # noqa: E402
from transformers.training_args import TrainingArguments  # noqa: E402
from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
import dataquality as dq
import os 
from dataquality.utils.hf_datasets import load_pandas_df
from torch.nn import Module
from typing import List,Dict


# Trainer callback for Huggingface transformers Trainer library
class DQCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that sends the logs to [Galileo](https://www.rungalileo.io/).
    """
    def __init__(self):
        self.helper_data =  dq.get_model_logger().logger_config.helper_data
        self._initialized = False




    def _clear_logger_config_helper_data(self) -> None:
        self.helper_data["embs"] = None
        self.helper_data["probs"] = None
        self.helper_data["logits"] = None


    def setup(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self._dq = dq
        # 🔭🌕 Galileo logging
        self._dq.init(task_type="text_classification", 
        project_name="text_classification_pytorch_hook_beta", 
        run_name=f"example_run_emotion_idx_1")
        #dq.init(
            #project=os.getenv("DQ_PROJECT", "huggingface"),
            #name=run_name,
            #**init_args
         #   )

        self._attach_hooks_to_model(model)
        eval_dataloader = kwargs["eval_dataloader"]
        train_dataloader = kwargs["train_dataloader"]
        self.tokenizer = kwargs["tokenizer"]
        
        # 🔭🌕 Galileo logging
        dq.log_dataset(load_pandas_df(train_dataloader.dataset), split="train")


        if getattr(eval_dataloader,"dataset",False):
            dq.log_dataset(load_pandas_df(eval_dataloader.dataset), split="validation")

        dq.set_labels_for_run(train_dataloader.dataset.features['label'].names)
        self._initialized = True



    def on_train_begin(self, args: TrainingArguments, state: TrainerState, model: Module, control: TrainerControl, **kwargs):
        if not self._initialized:
            self.setup(state, model , **kwargs)
        self._dq.set_split("training") # 🔭🌕 Galileo logging

    def on_epoch_begin(self,  args: TrainingArguments, state: TrainerState,  **kwargs):
        self._dq.set_epoch(state.epoch) # 🔭🌕 Galileo logging

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self._dq.set_split("validation") # 🔭🌕 Galileo logging

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        pass

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Perform a training step on a batch of inputs.
        """
        #Log only if embedding exists
        if self.last_step["embs"] != None and self.last_step["logits"] != None  and self.last_step["ids"] != None:
            # 🔭🌕 Galileo logging
            self._dq.log_model_outputs(**self.last_step)

    def _attach_hooks_to_model(self,model: Module):
        next(model.children()).register_forward_hook(self._embedding_hook)
        model.register_forward_hook(self._logit_hook)


    def _embedding_hook(self,model: Module, input, output):
        output_detached = output.last_hidden_state.detach()
        self.last_step["embs"] = output_detached[:,0]


    def _logit_hook(self,model: Module, model_input, model_output):
        #log the output logits
        self.last_step["logits"] = model_output.logits.detach()
 
    #workaround to save the idx
    #TODO: find cleaner way
    #ADD arguments
    def _collate_fn(self,rows:List[Dict]):
        #in: ['text', 'label', 'idx', 'input_ids', 'attention_mask']
        indices = [row.get("idx") for row in rows]
        self.last_step["ids"] = indices
        cleaned_rows = [{
            "label":row.get("label"),
            "input_ids":row.get("input_ids"),
            "attention_mask":row.get("attention_mask"),
                        } for row in rows]

        #out: ['label', 'input_ids', 'attention_mask']
        return DataCollatorWithPadding(self.tokenizer)(cleaned_rows)