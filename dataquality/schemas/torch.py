from typing import List, Optional, Tuple, Union

from torch import Tensor
from torch.nn import Module

DimensionSlice = Union[int, slice, Tensor, List, Tuple]
InputDim = Optional[Union[str, DimensionSlice]]
Layer = Optional[Union[Module, str]]
