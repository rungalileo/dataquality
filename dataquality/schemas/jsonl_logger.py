from typing import Dict, List, Optional

from pydantic import BaseModel, validator
from pydantic.types import StrictFloat, StrictInt, StrictStr

from dataquality.schemas.split import Split


class JsonlInputLogItem(BaseModel):
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


class JsonlOutputLogItem(BaseModel):
    id: StrictInt
    split: Split
    epoch: StrictInt
    emb: List[StrictFloat]
    prob: List[StrictFloat]
    pred: Optional[StrictStr] = None
