from enum import Enum
from typing import Any, Callable, Dict
from fastai.callback.core import Callback
from functools import partial

from torch.nn import Module
import dataquality as dq
from dataquality.schemas.split import Split
from dataquality.utils.fastai import convert_dl_to_df, forward_hook_with_store


class FastAiKeys(Enum):
    dl_next_idx_ids = "dl_next_idx_ids"
    dl_curent_batch_ids = "dl_curent_batch_ids"
    model_input = "model_input"
    model_output = "model_output"
    ids = "ids"


class IdxLogPatch:
    """
    Patch the DataLoader to store the indices of the batches.
    For example:
    self.dl.get_idxs = IdxLogPatch(self.dl.get_idxs, self.idx_log)
    """

    def __init__(self, old_func: Callable, store: Dict[str, Any]):
        """
        Patch the DataLoader to store the indices of the batches.
        For example:
        self.dl.get_idxs = IdxLogPatch(self.dl.get_idxs, self.idx_log)
        :param old_func: The original function to patch.
        :param store: The store to store the indices in.
        """
        self.old_func = old_func
        self.store = store
        self.store[FastAiKeys.dl_next_idx_ids] = []

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        Call the original function and store the indices.
        :param args: The arguments to pass to the original function.
        :param kwargs: The keyword arguments to pass to the original function.

        """
        res = self.old_func(*args, **kwargs)
        if res:
            self.store[FastAiKeys.dl_next_idx_ids].append(res)
        return res


class DQFastAiCallback(Callback):
    """
    Callback to log the model training for data quality.
    """

    current_model_outputs = {}
    idx_outputs = {FastAiKeys.dl_curent_batch_ids: []}
    hook = None
    layer = None
    disable_dq = False
    is_initialized = False
    labels = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Callback to log the model training for data quality.
        :param layer: Classifier layer with embeddings as input and logits as output.
        :param disable_dq: Disable data quality logging.
        :param args: The arguments to pass to the super class.
        :param kwargs: The keyword arguments to pass to the super class.
        """
        super().__init__(*args, **kwargs)
        self.disable_dq = kwargs.pop("disable_dq", False)
        self.layer = kwargs.pop("layer", None)
        self.labels = kwargs.pop("labels", None)
        assert (
            self.labels is not None
        ), "Labels must be provided. DQFastAiCallback(labels=['negative','positive'])"

        if not self.disable_dq:
            dq.init(*args, **kwargs)

    def get_layer(self) -> Module:
        """
        Get the classifier layer, which inputs and outputs will be logged
        (embeddings and logits).
        :return: The classifier layer.
        """
        if self.layer is None:
            # Get the last layer of the model
            return list(list(self.model.children())[-1].children())[-1]
        else:
            return self.layer

    def before_epoch(self) -> None:
        """
        Sets the epoch in data quality.
        """
        if not self.disable_dq:
            dq.set_epoch(self.epoch)

    def before_train(self) -> None:
        """
        Sets the split in data quality and registers the classifier layer hook.
        """
        if not self.disable_dq:
            dq.set_split(Split.train)
            self.wrap_indices()
            if self.is_initialized:
                return
            self.register_hooks()
            self.log_data()
            self.is_initialized = True

    def wrap_indices(self) -> None:
        """
        Wraps the get_idxs function of the dataloader to store the indices.
        """
        if not isinstance(self.dl.get_idxs, IdxLogPatch):
            self.dl.get_idxs = IdxLogPatch(self.dl.get_idxs, self.idx_outputs)

    def before_validate(self):
        """
        Sets the split in data quality and registers the classifier layer hook.
        """
        if not self.disable_dq:
            dq.set_split(Split.validation)
        self.wrap_indices()

    def log_data(self):
        """
        Uploads data to galileo and removes the classifier layer hook.
        """
        if not self.disable_dq:
            dq.set_labels_for_run(self.labels)
            train_dl, valid_dl = self.dls
            dq.log_image_dataset(  # TODO: add support for other datasets
                convert_dl_to_df(train_dl),
                imgs_colname="image",
                imgs_location_colname="path",
                split=Split.train,
            )
            dq.log_image_dataset(
                convert_dl_to_df(valid_dl),
                imgs_colname="image",
                imgs_location_colname="path",
                split=Split.validation,
            )

    def after_fit(self):
        """
        Uploads data to galileo and removes the classifier layer hook.
        """
        if not self.disable_dq:
            dq.finish()
        try:
            self.h.remove()
        except Exception:
            pass

    def before_batch(self):
        """
        Clears the model outputs log.
        """
        self.model_outputs_log.clear()

    def after_batch(self) -> None:
        """
        Logs the model outputs.
        """
        # If the current batch is empty, get the next batch ids from the store
        if len(self.idx_outputs[FastAiKeys.dl_curent_batch_ids]) == 0:
            self.idx_outputs[FastAiKeys.dl_curent_batch_ids] = self.idx_outputs[
                FastAiKeys.dl_next_idx_ids
            ].pop(0)
        # Get the current batch size
        bs_len = len(self.model_outputs_log[FastAiKeys.model_input])
        # Store the current batch ids by trimming the stored ids by
        # the batch size length
        self.model_outputs_log[FastAiKeys.ids] = self.idx_outputs[
            FastAiKeys.dl_curent_batch_ids
        ][:bs_len]
        self.current_idx = self.current_idx[bs_len:]
        # Log the model outputs
        embs = self.model_outputs_log[FastAiKeys.model_input][0].detach().cpu().numpy()
        logits = self.model_outputs_log[FastAiKeys.model_output].detach().cpu().numpy()
        ids = self.model_outputs_log[FastAiKeys.ids]
        log = not self.disable_dq
        equal_len = len(embs) == len(logits) and len(ids) == len(embs)
        if not equal_len:
            print("length not equal", len(logits), len(ids), len(embs))
        if log and equal_len:
            dq.log_model_outputs(embs=embs, logits=logits, ids=ids)

    def register_hooks(self) -> None:
        """
        Registers the classifier layer hook.
        """
        h = None
        if not self.hook:
            forward_hook = partial(forward_hook_with_store, self.model_outputs_log)
            h = self.get_layer().register_forward_hook(forward_hook)
            self.hook = h
        return h

    def __call__(self, event_name, *args: Any, **kwargs: Any):
        """
        Calling the callback for the given event.
        :param event_name: The event name (before_train, before_batch, after_fit... etc)
        """
        if hasattr(self, event_name):
            getattr(self, event_name)()
