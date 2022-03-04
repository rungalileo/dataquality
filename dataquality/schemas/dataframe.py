from pydantic import BaseModel
from vaex.dataframe import DataFrame


class BaseLoggerInOutFrames(BaseModel):
    prob: DataFrame
    emb: DataFrame
    data: DataFrame

    class Config:
        arbitrary_types_allowed = True
