from typing import List

from pydantic import BaseModel
from pydantic.types import StrictFloat, StrictInt, StrictStr

from dataquality.schemas.logger import LoggerMode


class JsonlTrainingInputItem(BaseModel):
    id: StrictInt
    text: StrictStr
    gold: StrictStr
    logger_mode: LoggerMode = LoggerMode.training


class JsonlTrainingOutputItem(BaseModel):
    id: StrictInt
    epoch: StrictInt
    emb: List[StrictFloat]
    prob: List[StrictFloat]
    logger_mode: LoggerMode = LoggerMode.training


class JsonlTrainingOutputItemLogged(JsonlTrainingOutputItem):
    pred: StrictInt


class JsonlValidationInputItem(BaseModel):
    id: StrictInt
    text: StrictStr
    gold: StrictStr
    logger_mode: LoggerMode = LoggerMode.validation


class JsonlValidationOutputItem(BaseModel):
    id: StrictInt
    epoch: StrictInt
    emb: List[StrictFloat]
    prob: List[StrictFloat]
    logger_mode: LoggerMode = LoggerMode.validation


class JsonlValidationOutputItemLogged(JsonlValidationOutputItem):
    pred: StrictInt


class JsonlTestInputItem(BaseModel):
    id: StrictInt
    text: StrictStr
    gold: StrictStr
    logger_mode: LoggerMode = LoggerMode.test


class JsonlTestOutputItem(BaseModel):
    id: StrictInt
    emb: List[StrictFloat]
    prob: List[StrictFloat]
    logger_mode: LoggerMode = LoggerMode.test


class JsonlTestOutputItemLogged(JsonlTestOutputItem):
    pred: StrictInt


class JsonlInferenceInputItem(BaseModel):
    id: StrictInt
    text: StrictStr
    logger_mode: LoggerMode = LoggerMode.inference


class JsonlInferenceOutputItem(BaseModel):
    id: StrictInt
    emb: List[StrictFloat]
    prob: List[StrictFloat]
    logger_mode: LoggerMode = LoggerMode.inference


class JsonlInferenceOutputItemLogged(JsonlInferenceOutputItem):
    pred: StrictInt
