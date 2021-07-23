from typing import List

from pydantic import BaseModel

from galileo_python.schemas.logger import LoggerMode


class JsonlTrainingInputItem(BaseModel):
    id: int
    text: str
    gold: str
    logger_mode: LoggerMode = LoggerMode.training


class JsonlTrainingOutputItem(BaseModel):
    id: int
    epoch: int
    emb: List[float]
    prob: List[float]
    logger_mode: LoggerMode = LoggerMode.training


class JsonlTrainingOutputItemLogged(JsonlTrainingOutputItem):
    pred: int


class JsonlValidationInputItem(BaseModel):
    id: int
    text: str
    gold: str
    logger_mode: LoggerMode = LoggerMode.validation


class JsonlValidationOutputItem(BaseModel):
    id: int
    epoch: int
    emb: List[float]
    prob: List[float]
    logger_mode: LoggerMode = LoggerMode.validation


class JsonlValidationOutputItemLogged(JsonlValidationOutputItem):
    pred: int


class JsonlTestInputItem(BaseModel):
    id: int
    text: str
    gold: str
    logger_mode: LoggerMode = LoggerMode.test


class JsonlTestOutputItem(BaseModel):
    id: int
    emb: List[float]
    prob: List[float]
    logger_mode: LoggerMode = LoggerMode.test


class JsonlTestOutputItemLogged(JsonlTestOutputItem):
    pred: int


class JsonlInferenceInputItem(BaseModel):
    id: int
    text: str
    logger_mode: LoggerMode = LoggerMode.inference


class JsonlInferenceOutputItem(BaseModel):
    id: int
    emb: List[float]
    prob: List[float]
    logger_mode: LoggerMode = LoggerMode.inference


class JsonlInferenceOutputItemLogged(JsonlInferenceOutputItem):
    pred: int
