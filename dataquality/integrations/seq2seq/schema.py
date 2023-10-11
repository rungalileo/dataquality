# TODO: create config schema for training, data, and generation
from dataclasses import dataclass


@dataclass
class AutoTrainingConfig:
    name: str
    input_col: str
    target_col: str


@dataclass
class AutoDataConfig:
    name: str
    input_col: str
    target_col: str


@dataclass
class AutoGenerationConfig:
    name: str
    input_col: str
    target_col: str
