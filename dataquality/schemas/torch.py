from typing import List, Optional, Tuple, Union

from torch import Tensor
from torch.nn import Module

EmbeddingDim = Union[int, slice, Tensor, List, Tuple]
Layer = Optional[Union[Module, str]]
