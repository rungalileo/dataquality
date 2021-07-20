from enum import Enum, unique
from typing import List

from pydantic import BaseModel


@unique
class LoggerMode(str, Enum):
    training = "training"
    validation = "validation"
    test = "test"
    inference = "inference"


class TrainingInputItem(BaseModel):
    id: int
    text: str
    gold: str
    logger_mode: LoggerMode = LoggerMode.training


class TrainingOutputItem(BaseModel):
    id: int
    epoch: int
    emb: List[float]
    prob: List[float]
    logger_mode: LoggerMode = LoggerMode.training


class TrainingOutputItemLogged(TrainingOutputItem):
    pred: int


class ValidationInputItem(BaseModel):
    id: int
    text: str
    gold: str
    logger_mode: LoggerMode = LoggerMode.validation


class ValidationOutputItem(BaseModel):
    id: int
    epoch: int
    emb: List[float]
    prob: List[float]
    logger_mode: LoggerMode = LoggerMode.validation


class ValidationOutputItemLogged(ValidationOutputItem):
    pred: int


class TestInputItem(BaseModel):
    id: int
    text: str
    gold: str
    logger_mode: LoggerMode = LoggerMode.test


class TestOutputItem(BaseModel):
    id: int
    emb: List[float]
    prob: List[float]
    logger_mode: LoggerMode = LoggerMode.test


class TestOutputItemLogged(TestOutputItem):
    pred: int


class InferenceInputItem(BaseModel):
    id: int
    text: str
    logger_mode: LoggerMode = LoggerMode.inference


class InferenceOutputItem(BaseModel):
    id: int
    emb: List[float]
    prob: List[float]
    logger_mode: LoggerMode = LoggerMode.inference


class InferenceOutputItemLogged(InferenceOutputItem):
    pred: int
