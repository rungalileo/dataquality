from enum import Enum, unique
from typing import List, Optional, Tuple, Union

from torch import Tensor
from torch.nn import Module

DimensionSlice = Union[int, slice, Tensor, List, Tuple]
InputDim = Optional[Union[str, DimensionSlice]]
Layer = Optional[Union[Module, str]]


@unique
class HelperData(str, Enum):
    """A collection of all default attributes across all loggers"""

    dqcallback = "dqcallback"
    signature_cols = "signature_cols"
    orig_collate_fn = "orig_collate_fn"
    model_outputs_store = "model_outputs_store"
    model = "model"
    hook_manager = "hook_manager"
    last_action = "last_action"
    patches = "patches"
    dl_next_idx_ids = "dl_next_idx_ids"
    batch = "batch"
    model_input = "model_input"
