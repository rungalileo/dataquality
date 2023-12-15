from collections import defaultdict
from typing import Any, Callable, DefaultDict, Dict, List, Optional, Set

from pydantic import BaseModel, ConfigDict, ValidationInfo, field_validator

from dataquality.schemas.condition import Condition
from dataquality.schemas.ner import TaggingSchema
from dataquality.schemas.split import Split


class BaseLoggerConfig(BaseModel):
    labels: Any = None
    tasks: Any = None
    observed_num_labels: Any = None
    observed_labels: Any = None
    tagging_schema: Optional[TaggingSchema] = None
    last_epoch: int = 0
    cur_epoch: Optional[int] = None
    cur_split: Optional[Split] = None
    cur_inference_name: Optional[str] = None
    training_logged: bool = False
    validation_logged: bool = False
    test_logged: bool = False
    inference_logged: bool = False
    exception: str = ""
    helper_data: Dict[str, Any] = {}
    input_data_logged: DefaultDict[str, int] = defaultdict(int)
    logged_input_ids: DefaultDict[str, Set] = defaultdict(set)
    idx_to_id_map: DefaultDict[str, List] = defaultdict(list)
    conditions: List[Condition] = []
    report_emails: List[str] = []
    ner_labels: List[str] = []
    int_labels: bool = False
    feature_names: List[str] = []
    metadata_documents: Set = set()  # A document is a large str > 1k chars < 10k chars
    finish: Callable = lambda: None  # Overwritten in Semantic Segmentation
    # True when calling `init` with a run that already exists
    existing_run: bool = False
    dataloader_random_sampling: bool = False
    remove_embs: bool = False

    model_config = ConfigDict(validate_assignment=True)

    def reset(self, factory: bool = False) -> None:
        """Reset all class vars"""
        self.__init__()  # type: ignore

    @field_validator("cur_split", mode="after")
    @classmethod
    def inference_sets_inference_name(cls, field_value: Split, validation_info: ValidationInfo) -> Split:
        values = validation_info.data
        if field_value == Split.inference:
            split_name = values.get("cur_inference_name")
            if not split_name:
                raise ValueError("Please specify inference_name when setting split to inference")
        return field_value


base_logger_config = BaseLoggerConfig()
