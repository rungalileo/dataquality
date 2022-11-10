from enum import Enum
from typing import List, Optional

from pydantic import BaseModel


class ConditionStatus(str, Enum):
    passed = "passed"
    failed = "failed"


class SplitConditionData(BaseModel):
    split: str
    inference_name: Optional[str]
    status: ConditionStatus
    link: Optional[str]
    ground_truth: float


class ReportConditionData(BaseModel):
    condition: str
    criteria: str
    splits: List[SplitConditionData]


class RunReportData(BaseModel):
    template: str
    email_subject: str
    project_name: str
    run_name: str
    created_at: str
    link: str
    conditions: List[ReportConditionData]
