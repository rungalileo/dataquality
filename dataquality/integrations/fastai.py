from enum import Enum
from functools import partial
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from fastai.callback.core import Callback
from fastai.data.load import DataLoader
from torch.nn import Module
from torch.utils.hooks import RemovableHandle

import dataquality
from dataquality import config
from dataquality.analytics import Analytics
from dataquality.clients.api import ApiClient
from dataquality.exceptions import GalileoException
from dataquality.loggers.logger_config.base_logger_config import BaseLoggerConfig
from dataquality.schemas.split import Split
from dataquality.utils.helpers import galileo_disabled

a = Analytics(ApiClient, config)
a.log_import("fastai")


class FastAiKeys(Enum):
    dataloader_indices = "dataloader_indices"
    idx_queue = "idx_queue"
    model_input = "model_input"
    model_output = "model_output"
    ids = "ids"


class _PatchDLGetIdxs:
    """
    Patch the DataLoader to store the indices of the batches.
    For example:
    self.dl.get_idxs = _PatchDLGetIdxs(self.dl.get_idxs, self.idx_log)
    """

    def __init__(self, old_func: Callable, store: Dict[FastAiKeys, Any]) -> None:
        """
        Patch the DataLoader to store the indices of the batches.
        For example:
        self.dl.get_idxs = IdxLogPatch(self.dl.get_idxs, self.idx_log)
        :param old_func: The original function to patch.
        :param store: The store to store the indices in.
        """
        self.logger_config = dataquality.get_model_logger().logger_config
        self.old_func = old_func
        self.store = store
        self.store[FastAiKeys.dataloader_indices] = {
            Split.training: [],
            Split.validation: [],
            Split.test: [],
        }

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        Call the original function and store the indices.
        :param args: The arguments to pass to the original function.
        :param kwargs: The keyword arguments to pass to the original function.
        """
        res = self.old_func(*args, **kwargs)
        if res:
            self.store[FastAiKeys.dataloader_indices][
                self.logger_config.cur_split
            ].append(res)
        return res


class FastAiDQCallback(Callback):
    """
    Dataquality logs the model embeddings and logtis to measure the quality
    of the dataset. Provide the label names and the classifier layer to log
    the embeddings and logits. If no classifier layer is provided,
    the last layer of the model will be used.
    Here is how to take the last layer of the model:
    `dqc = DataqualityCallback(labels=['negative','positive'], layer=model.fc)`
    End to end example:
    .. code-block:: python

        from fastai.vision.all import *
        from fastai.callback.galileo import DataqualityCallback
        path = untar_data(URLs.PETS)/'images'
        image_files = get_image_files(path)#[:107]
        label_func = lambda x: x[0].isupper()
        dls = ImageDataLoaders.from_name_func(
            path, image_files, valid_pct=0.2,
            label_func=label_func, item_tfms=Resize(224),
            num_workers=1, drop_last=False)
        learn = vision_learner(dls, 'resnet34', metrics=error_rate)
        dqc = DataqualityCallback(labels=["nocat","cat"])
        learn.add_cb(dqc)
        learn.fine_tune(2)

    """

    hook = None
    is_initialized = False
    labels = None
    model_outputs_log: Dict[FastAiKeys, Any]
    current_idx: List[int]
    logger_config: BaseLoggerConfig
    idx_store: Dict[FastAiKeys, Any]
    disable_dq: bool = False

    def __init__(
        self,
        layer: Any = None,
        finish: bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Dataquality logs the model embeddings and logits to measure the quality
        of the dataset. This helps to find mislabelled samples in a
        data centric approach.
        :param layer: Classifier layer with embeddings as input and logits as output.
        :param finish: Upload after training is complete
        :param disable_dq: Disable data quality logging.
        """
        super().__init__(*args, **kwargs)
        a.log_function("fastai/callback")
        self.disable_dq = galileo_disabled()
        self.finish = finish
        self.layer = layer
        self.model_outputs_log = {}
        self.current_idx = []
        self.idx_store = {FastAiKeys.idx_queue: []}
        self.counter = 0
        if config.task_type not in ["text_classification", "image_classification"]:
            raise GalileoException(
                f"task_type {str(config.task_type)} is not supported yet.\
                    Try dq.init(task_type='image_classification' or\
                          'text_classification')"
            )

        if not self.disable_dq:
            self.logger_config = dataquality.get_model_logger().logger_config

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
        if not self.disable_dq:
            dataquality.set_epoch(self.epoch)

    def before_fit(self) -> None:
        assert (
            self.dls.drop_last is not None and not self.dls.drop_last
        ), "DataLoader must be initialized with drop_last=False"
        if self.is_initialized or self.disable_dq:
            return
        self.wrap_indices()
        self.register_hooks()

        self.is_initialized = True

    def before_train(self) -> None:
        """
        Sets the split in data quality and registers the classifier layer hook.
        """
        if self.disable_dq:
            return
        dataquality.set_split(dataquality.schemas.split.Split.train)
        self.wrap_indices()
        if self.is_initialized:
            return
        self.register_hooks()

        self.is_initialized = True

    def wrap_indices(self) -> None:
        """
        Wraps the get_idxs function of the dataloader to store the indices.
        """
        if not hasattr(self, "dl"):
            print("Dataloader not found")
            return
        if not isinstance(self.dl.get_idxs, _PatchDLGetIdxs):
            print("Wrapping dataloader")
            self.dl.get_idxs = _PatchDLGetIdxs(self.dl.get_idxs, self.idx_store)

    def after_validate(self) -> None:
        if self.disable_dq:
            return
        dataquality.set_split(dataquality.schemas.split.Split.train)

    def before_validate(self) -> None:
        """
        Sets the split in data quality and registers the classifier layer hook.
        """
        self.wrap_indices()
        if self.disable_dq:
            return
        dataquality.set_split(dataquality.schemas.split.Split.validation)
        self.idx_store[FastAiKeys.idx_queue] = []

    def after_fit(self) -> None:
        """
        Uploads data to galileo and removes the classifier layer hook.
        """
        if (self.n_epoch - 1) == self.epoch:
            self.counter += 1

        if self.counter != 2:
            return
        print("Finishing dataquality")
        try:
            self.h.remove()
        except Exception:
            pass
        if self.finish:
            dataquality.finish()

    def before_batch(self) -> None:
        """
        Clears the model outputs log.
        """
        self.model_outputs_log.clear()

    def after_pred(self) -> None:
        """
        Logs the model outputs.
        """
        # Get the current batch size
        bs_len = len(self.model_outputs_log[FastAiKeys.model_output])
        # Store the current batch ids by trimming the stored ids by
        # the batch size length
        cur_split = self.logger_config.cur_split
        indices = self.idx_store[FastAiKeys.dataloader_indices][cur_split][-1][
            :bs_len
        ].copy()
        idx_store = self.idx_store[FastAiKeys.dataloader_indices][cur_split][-1][
            bs_len:
        ]
        self.idx_store[FastAiKeys.dataloader_indices][cur_split][-1] = idx_store
        try:
            if cur_split is not None:
                id_map = self.logger_config.idx_to_id_map[cur_split]
                ids = np.array([id_map[i] for i in indices])
            else:
                print("current split needs to be set")
                return
        except Exception as e:
            print("cur_split error", cur_split, e)
            return
        # Log the model outputs
        embs = self.model_outputs_log[FastAiKeys.model_input][0].detach().cpu().numpy()
        logits = self.model_outputs_log[FastAiKeys.model_output].detach().cpu().numpy()
        equal_len = len(embs) == len(logits) == len(ids) == len(embs)
        if not equal_len:
            raise GalileoException(
                f"Logging failed: length not equal.\
                    logits: {len(logits)},ids: {len(ids)},\
 embs: {len(embs)}"
            )
        if self.disable_dq or not equal_len:
            return

        dataquality.log_model_outputs(embs=embs, logits=logits, ids=ids)

    def register_hooks(self) -> Optional[RemovableHandle]:
        """
        Registers the classifier layer hook.
        """
        h = None
        if not self.hook:
            forward_hook = partial(self.forward_hook_with_store, self.model_outputs_log)
            h = self.get_layer().register_forward_hook(forward_hook)
            self.hook = h
        return h

    def forward_hook_with_store(
        self,
        store: Dict[FastAiKeys, Any],
        layer: Module,
        model_input: Any,
        model_output: Any,
    ) -> None:
        """
        Forward hook to store the output of a layer.
        :param store: Dictionary to store the output in.
        :param layer: Layer to store the output of.
        :param model_input: Input to the model.
        :param model_output: Output of the model.
        :return: None
        """
        store[FastAiKeys.model_input] = model_input
        store[FastAiKeys.model_output] = model_output


def convert_img_dl_to_df(dl: DataLoader, x_col: str = "image") -> pd.DataFrame:
    """
    Converts a fastai DataLoader to a pandas DataFrame.
    :param dl: Fast ai DataLoader to convert.
    :param x_col: Name of the column to use for the x values, for example image.
    :return: Pandas DataFrame with the data from the DataLoader.
    """
    a.log_function("fastai/convert_img_dl_to_df")
    additional_data = {}
    if x_col == "image":
        additional_data["text"] = dl.items
    x, y = [], []
    for x_item, y_item in dl.dataset:
        x.append(x_item)
        y.append(int(y_item))
    ids = dl.vocab.o2i.keys()
    if len(ids) == 2 and isinstance(next(iter(ids)), bool):
        ids = dl.dataset.splits[dl.dataset.split_idx]
    df = pd.DataFrame({"id": ids, x_col: x, "label": y, **additional_data})
    del additional_data, x, y
    return df


def convert_tab_dl_to_df(
    dl: DataLoader, x_col: str = "text", y_col: str = "label"
) -> pd.DataFrame:
    """
    Converts a fastai DataLoader to a pandas DataFrame.
    :param dl: Fast ai DataLoader to convert.
    :param x_col: Name of the column to use for the x values, for example text.
    :param y_col: Name of the column to use for the y values, for example label.
    :return: Pandas DataFrame with the data from the DataLoader.
    """
    a.log_function("fastai/convert_tab_dl_to_df")
    df = dl.items.copy()
    df = df.rename(columns={x_col: "text", y_col: "label"})
    if "id" not in df.columns:
        df["id"] = df.index
    return df
