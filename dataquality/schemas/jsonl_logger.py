from typing import Any, Dict, List, Optional, Union

import numpy as np
from pydantic import BaseModel, validator
from pydantic.types import StrictFloat, StrictInt, StrictStr

from dataquality.schemas import __data_schema_version__
from dataquality.schemas.split import Split

try:
    from torch import Tensor

    USING_TORCH = True
except ImportError:
    USING_TORCH = False


class BaseLogItem(BaseModel):
    data_schema_version: int = __data_schema_version__

    @validator("data_schema_version", always=True)
    def validate_version(cls: BaseModel, v: int) -> int:
        if v != __data_schema_version__:
            raise ValueError(
                "You cannot change the data_schema_version! This is for "
                "internal use only."
            )
        return v


class JsonlInputLogItem(BaseLogItem):
    id: StrictInt
    split: Split
    text: StrictStr
    gold: Optional[StrictStr] = None

    @validator("gold", always=True)
    def gold_for_split(cls: BaseModel, v: str, values: Dict) -> Optional[str]:
        if v is not None and values["split"] == Split.inference:
            raise ValueError("gold should not be defined for inference!")
        if v is None and values["split"] != Split.inference:
            raise ValueError(f"gold must be defined for {values['split']}!")
        return v


class JsonlOutputLogItem(BaseLogItem):
    id: StrictInt
    split: Split
    epoch: StrictInt
    emb: List[StrictFloat]
    prob: List[StrictFloat]
    pred: Optional[StrictStr] = None

    @validator("emb", "prob", always=True, pre=True)
    def validate_embedding_shape(
        cls, v: Union[List[StrictFloat], Any]
    ) -> List[StrictFloat]:
        if USING_TORCH and isinstance(v, Tensor):  # Validate the tensor shape is 1D
            if len(v.shape) != 1:
                raise ValueError(f"Tensor shape must be 1D, but got {v.shape}")
            return v.detach().numpy().tolist()
        elif isinstance(v, np.ndarray):
            return v.tolist()
        return v
