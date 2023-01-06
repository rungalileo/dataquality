import gc
from functools import wraps
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Union

from datasets import Dataset
from torch.nn import Module
from torch.utils.data.dataloader import _BaseDataLoaderIter
from torch.utils.data.sampler import Sampler
from transformers import Trainer
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

import dataquality as dq
from dataquality.exceptions import GalileoException
from dataquality.integrations.torch import TorchBaseInstance
from dataquality.schemas.split import Split
from dataquality.schemas.torch import DimensionSlice, InputDim, Layer
from dataquality.utils.torch import ModelHookManager
from dataquality.utils.transformers import (
    add_id_to_signature_columns,
    remove_id_collate_fn_wrapper,
)


def patch_dataloaders(store: Dict, reset_indices: bool = True) -> None:
    def wrap_next_index(func: Callable, key: str = "ids") -> Callable:
        @wraps(func)
        def patched_next_index(*args: Any, **kwargs: Any) -> Any:
            print("patched_next_index", args, kwargs)
            indices = func(*args, **kwargs)
            if indices and key in store:
                if reset_indices and store.get("last_action") == "pop":
                    store[key] = []
                store[key].append(indices)
                store["last_action"] = "append"

            return indices

        return patched_next_index

    if hasattr(_BaseDataLoaderIter, "_patched"):
        # logger warning if already patched
        print("BaseDataLoaderIter already patched")
        return
    if "patches" not in store:
        store["patches"] = []

    store["patches"].append({"class": _BaseDataLoaderIter, "attr": "_next_index"})
    setattr(_BaseDataLoaderIter, "_old__next_index", _BaseDataLoaderIter._next_index)
    setattr(
        _BaseDataLoaderIter,
        "_next_index",
        wrap_next_index(_BaseDataLoaderIter._next_index, "ids"),
    )
    setattr(_BaseDataLoaderIter, "_patched", True)


def unpatch(store: Dict) -> None:
    # unpatch all instances and classes
    # starting with all classes
    for patch in store.get("patches", []):
        print("unpatching", patch["class"])
        if hasattr(patch["class"], "_patched"):
            print("unpatching", patch["class"], patch["attr"])
            setattr(
                patch["class"],
                patch["attr"],
                getattr(patch["class"], f"_old_{patch['attr']}"),
            )
            delattr(patch["class"], f"_old_{patch['attr']}")
            delattr(patch["class"], "_patched")
        # then all instances
        for obj in gc.get_objects():
            if isinstance(obj, patch["class"]) and hasattr(obj, "_patched"):
                print("unpatching", obj, patch["attr"])
                setattr(obj, patch["attr"], getattr(obj, f"old_{patch['attr']}"))
                delattr(obj, f"old_{patch['attr']}")
                delattr(obj, "_patched")


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
        classifier_layer: Layer = None,
        embedding_dim: InputDim = None,
        logits_dim: InputDim = None,
        embedding_fn: Optional[Any] = None,
        logits_fn: Optional[Any] = None,
    ) -> None:
        # Access the dq logger helper data
        self.helper_data = dq.get_model_logger().logger_config.helper_data
        self._initialized = False
        # Hook manager for attaching hooks to the model
        self.hook_manager = ModelHookManager()
        self.classifier_layer = classifier_layer
        self._set_dimensions(embedding_dim, logits_dim)
        self.embedding_fn = embedding_fn
        self.logits_fn = logits_fn

    def _clear_logger_config_helper_data(self) -> None:
        self.helper_data.clear()

    def _do_log(self) -> None:
        # Log only if embedding exists
        assert self.helper_data.get("embs") is not None, GalileoException(
            "Embedding passed to the logger can not be logged"
        )
        assert self.helper_data.get("logits") is not None, GalileoException(
            "Logits passed to the logger can not be logged"
        )
        assert self.helper_data.get("ids") is not None, GalileoException(
            "Did you map IDs to your dataset before watching the model? You can run:\n"
            "`ds= dataset.map(lambda x, idx: {'id': idx}, with_indices=True)`\n"
            "id (index) column is needed in the dataset for logging"
        )

        # 🔭🌕 Galileo logging
        dq.log_model_outputs(**self.helper_data)
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
        self._attach_hooks_to_model(model, self.classifier_layer)
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

    def _attach_hooks_to_model(self, model: Module, layer: Layer) -> None:
        """
        Method to attach hooks to the model by using the hook manager
        :param model: Model
        :param model: pytorch model layer to attach hooks to
        :return: None
        """

        self.hook_manager.attach_embedding_hook(model, self._classifier_hook, layer)


def watch(
    trainer: Trainer,
    classifier_layer: Union[str, Module] = "classifier",
    embedding_dim: Optional[DimensionSlice] = None,
    logits_dim: Optional[DimensionSlice] = None,
    embedding_fn: Callable = None,
    logits_fn: Callable = None,
) -> None:
    # store = {}
    # patch_dataloaders(store)
    # The columns needed for the forward process
    signature_cols = add_id_to_signature_columns(trainer)

    # Callback which we add to the trainer
    dqcallback = DQCallback(
        classifier_layer=classifier_layer,
        embedding_dim=embedding_dim,
        logits_dim=logits_dim,
        embedding_fn=embedding_fn,
        logits_fn=logits_fn,
    )
    # We wrap the data collator to remove the id column
    trainer.data_collator = remove_id_collate_fn_wrapper(
        trainer.data_collator, signature_cols, dqcallback.helper_data
    )
    trainer.add_callback(dqcallback)


class BatchSamplerWithLogging(Sampler[List[int]]):
    r"""Wraps another sampler to yield a mini-batch of indices.

    Args:
        sampler (Sampler or Iterable): Base sampler. Can be any iterable object
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``

    Example:
        >>> list(BatchSampler(SequentialSampler(range(10)),
                batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3,
        drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    # batch_sampler = BatchSamplerWithLogging(
    #   RandomSampler(torch.arange(len(a))), 3, False, store)

    def __init__(
        self,
        sampler: Union[Sampler[int], Iterable[int]],
        batch_size: int,
        drop_last: bool,
        store: Dict,
    ) -> None:
        # Since collections.abc.Iterable does not check for `__getitem__`, which
        # is one way for an object to be an iterable, we don't do an `isinstance`
        # check here.
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size <= 0
        ):
            raise ValueError(
                "batch_size should be a positive integer value, "
                "but got batch_size={}".format(batch_size)
            )
        if not isinstance(drop_last, bool):
            raise ValueError(
                "drop_last should be a boolean value, but got "
                "drop_last={}".format(drop_last)
            )
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.store = store

    def __iter__(self) -> Iterator[List[int]]:
        # Implemented based on the benchmarking in:
        # https://github.com/pytorch/pytorch/pull/76951
        if self.drop_last:
            sampler_iter = iter(self.sampler)
            while True:
                try:
                    batch = [next(sampler_iter) for _ in range(self.batch_size)]
                    self.store["sampler_ids"].append(batch)
                    self.store["sampler_ids"].append(batch)
                    yield batch
                except StopIteration:
                    break
        else:
            batch = [0] * self.batch_size
            idx_in_batch = 0
            for idx in self.sampler:
                batch[idx_in_batch] = idx
                idx_in_batch += 1
                if idx_in_batch == self.batch_size:
                    self.store["sampler_ids"].append(batch)
                    yield batch
                    idx_in_batch = 0
                    batch = [0] * self.batch_size
            if idx_in_batch > 0:
                self.store["sampler_ids"].append(batch)
                yield batch[:idx_in_batch]
