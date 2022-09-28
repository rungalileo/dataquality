import inspect
from typing import Any, Dict, List

from torch.nn import Module
from transformers.data.data_collator import DataCollatorWithPadding
from transformers.trainer_callback import TrainerCallback  # noqa: E402
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments  # noqa: E402

from datasets import Dataset 


import dataquality as dq
from dataquality.exceptions import GalileoException
from dataquality.schemas.split import Split
from dataquality.utils.hf_datasets import load_pandas_df


# Imports for the hook manager
from queue import PriorityQueue
from queue import Queue



# Trainer callback for Huggingface transformers Trainer library
class DQCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that sends the logs to [Galileo](https://www.rungalileo.io/).
    """

    def __init__(self) -> None:
        self.helper_data = dq.get_model_logger().logger_config.helper_data
        self._initialized = False
        self.hook_manager = HookManager()

    def _clear_logger_config_helper_data(self) -> None:
        self.helper_data["embs"] = None
        self.helper_data["probs"] = None
        self.helper_data["logits"] = None

    def on_init_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of the initialization of the [`Trainer`].
        """
        pass
        


    def setup(
        self, args: TrainingArguments, state: TrainerState, model: Module, kwargs: Any
    ) -> None:
        self._dq = dq
        self._attach_hooks_to_model(model)
        eval_dataloader = kwargs["eval_dataloader"]
        train_dataloader = kwargs["train_dataloader"]
        self.tokenizer = kwargs["tokenizer"]

        # 🔭🌕 Galileo logging
        assert "id" in train_dataloader.dataset.column_names, GalileoException(
                "id column is needed for logging"
            )
        dq.log_dataset(load_pandas_df(train_dataloader.dataset),split=Split.train) #id=train_dataloader.dataset["idx"]
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

        assert self.helper_data.get("embs") is not None,GalileoException("Embedding passed to the logger can not be logged")
        assert self.helper_data.get("logits") is not None,GalileoException("Logits passed to the logger can not be logged")
        assert self.helper_data.get("ids") is not None,GalileoException("id column missing in dataset (needed to map rows to the indices/ids)")

        # 🔭🌕 Galileo logging
        self._dq.log_model_outputs(**self.helper_data)

    def _attach_hooks_to_model(self, model: Module) -> None:
      self.hook_manager.attach_embedding_hook(model,None,self._embedding_hook)
      self.hook_manager.attach_hook(model,self._logit_hook)
      
    def _embedding_hook(
        self, model: Module, model_input: Any, model_output: Any
    ) -> None:
        #TODO EMBEDDING CHECK
        #TODO optional number embedding check
        if hasattr(model_output,"last_hidden_state"):
            output_detached = model_output.last_hidden_state.detach()
        else:
            output_detached = model_output.detach()
        if len(output_detached.shape) == 3:
          print("initial_embedding shape",output_detached.shape)
          output_detached = output_detached[:, 0]
        self.helper_data["embs"] = output_detached[:, 0]

    def _logit_hook(self, model: Module, model_input: Any, model_output: Any) -> None:
        # log the output logits
        if hasattr(model_output,"logits"):
          logits = model_output.logits
        else:
          logits = model_output
        self.helper_data["logits"] = logits.detach()

    def add_id_col(self, dataset:Dataset) -> None:
        dataset  = dataset.add_column("id",list(range(len(dataset))))
        return dataset


class HookManager:
  """
  Manages hooks for models. Has the ability to find the layer automatically.
  Otherwise the layer or the layer name needs to be provided.
  """
  hooks = []

  def scoring_algorithm(self, layer_pos,level,model_name,layer_name):
    """
    The higher the score the more likely it is the embedding layer
    inputs are:
    layer_pos is the number of the layer
    level is the depth of the layer
    model_name is the name of the class
    layer_name is the name of the layer
    """
    score = 0
    embed = False
    if "embed" in layer_name.lower():
      embed = True
      score +=1.5
    if "embed" in model_name.lower():
      embed = True
      score +=1
    if "embedding" == model_name.lower():
      embed = True
      score +=1/8
    if "embedding" == layer_name.lower():
      embed = True
      score +=1/8
    # workaround to reduce impact of level / position scoriing
    if embed and level  / 5.5 > score:
      score -= level  / 7.5
    elif embed:
      score = score - level /  5.5
    if embed:
      score -=  layer_pos/2.53
    # to avoid collision in the priorityqueue 
    # TODO: test with a model that doesn't have an embedding layer
    return score - (level*10 + layer_pos) /1000


  def get_embedding_layer_auto(self, model):
    """
    Use the scoring algorithm in our breadth first search on the model.
    """
    # keeps track of model layers
    queue: Queue = Queue()
    # the start is the name / children of the model + the current layer level
    start = (model.named_children(),0)
    queue.put(start)
    #find the top choices for the embeddinglayer
    embeddings_layers = PriorityQueue()

    while not queue.empty():
      named_children,level = queue.get()
      layer_pos = 0
      for layer_name,layer_model in named_children:
        model_name = layer_model._get_name()
        score = self.scoring_algorithm(layer_pos,level,model_name,layer_name)
        if score > 0:
          #negative score because priorityqueue is reverse
          embeddings_layers.put((-score,(layer_model,level)))
        queue.put((layer_model.named_children(),level+1))
        layer_pos+=1
    print("Complete")
    el = embeddings_layers.get()
    return el[1]


  def get_embedding_layer_by_name(self, model,name):
    """
    Iterate over each layer and stop once the the layer name matches
    """
    queue: Queue = Queue()
    start = model.named_children()
    queue.put(start)
    while not queue.empty():
      named_children = queue.get()
      for layer_name,layer_model in named_children:
        if layer_name == name:
          return layer_model
        model_name = layer_model._get_name()
        queue.put(layer_model.named_children())
    # TODO: raise found layers
    # TODO: use galileo exception
    raise "Layer could not be found, make sure to check capitalization"

  def attach_embedding_hook(self,model,model_layer=None,embedding_hook=print):
    """attach hook and save it in our hook list"""
    if model_layer == None:
      selected_layer = self.get_embedding_layer_auto(model)
    elif type(model_layer) == str:
      selected_layer = self.get_embedding_layer_by_name(model_layer)
    else:
      selected_layer = model_layer
    h = self.attach_hook(selected_layer,embedding_hook)
    return h

  def attach_hook(self,selected_layer,hook):
    h = selected_layer.register_forward_hook(hook)
    self.hooks.append(h)
    return h

  def remove_hook(self):
    for h in self.hooks:
      h.remove()


def add_signature_columns(trainer):
  if trainer._signature_columns is None:
    # Inspect model forward signature to keep only the arguments it accepts.
    signature = inspect.signature(trainer.model.forward)
    trainer._signature_columns = list(signature.parameters.keys())
    # Labels may be named label or label_ids, the default data collator handles that.
    trainer._signature_columns += ["label", "label_ids"]
  if "id" not in trainer._signature_columns:
    trainer._signature_columns.append("id")

def remove_id_collate(function,store):
  """Removes the id from dict and passes it on.
    Simulates our logging of the id"""
  def remove_id(rows):
      "wrapper function"
      store["ids"] = [row.pop("id",None) for row in rows]
      return function(rows)

  return remove_id


def watch(trainer):
  dqcallback = DQCallback()
  add_signature_columns(trainer)
  trainer.data_collator = remove_id_collate(trainer.data_collator,dqcallback.helper_data) 
  trainer.add_callback(dqcallback)

  