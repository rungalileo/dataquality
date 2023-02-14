import os
import shutil
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Optional, Set

from pydantic import BaseModel, validator

from dataquality.schemas.condition import Condition
from dataquality.schemas.ner import TaggingSchema
from dataquality.schemas.split import Split
from dataquality.utils.constants import ConfigData

EXC_DIR = f"{ConfigData.DEFAULT_GALILEO_CONFIG_DIR}/exceptions"


class BaseLoggerConfig(BaseModel):
    labels: Any = None
    tasks: Any = None
    observed_num_labels: Any = None
    observed_labels: Any = None
    tagging_schema: Optional[TaggingSchema]
    last_epoch: int = 0
    cur_epoch: Optional[int]
    cur_split: Optional[Split]
    cur_inference_name: Optional[str]
    training_logged: bool = False
    validation_logged: bool = False
    test_logged: bool = False
    inference_logged: bool = False
    _exception: str = ""
    helper_data: Dict[str, Any] = {}
    input_data_logged: DefaultDict[str, int] = defaultdict(int)
    logged_input_ids: DefaultDict[str, Set] = defaultdict(set)
    idx_to_id_map: DefaultDict[str, List] = defaultdict(list)
    conditions: List[Condition] = []
    report_emails: List[str] = []
    ner_labels: List[str] = []
    int_labels: bool = False
    feature_names: List[str] = []

    class Config:
        validate_assignment = True

    def get_exception(self, run_id: str) -> str:
        exc_file = f"{EXC_DIR}/{run_id}.txt"
        if os.path.isfile(exc_file):
            with open(exc_file, "r") as f:
                return f.read()
        return ""

    def set_exception(self, run_id: str, exc: str) -> None:
        exc_file = f"{EXC_DIR}/{run_id}.txt"
        if not os.path.isdir(EXC_DIR):
            os.makedirs(EXC_DIR, exist_ok=True)
        with open(exc_file, "w") as f:
            f.write(exc)

    def reset(self, factory: bool = False) -> None:
        """Reset all class vars"""
        self.__init__()  # type: ignore
        if os.path.exists(EXC_DIR):
            shutil.rmtree(EXC_DIR)

    @validator("cur_split")
    def inference_sets_inference_name(
        cls, field_value: Split, values: Dict[str, Any]
    ) -> Split:
        if field_value == Split.inference:
            assert values.get(
                "cur_inference_name"
            ), "Please specify inference_name when setting split to inference"
        return field_value


base_logger_config = BaseLoggerConfig()
