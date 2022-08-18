from enum import Enum, unique
from typing import Any, Dict, Optional

from pydantic import UUID4, BaseModel, StrictInt, StrictStr, validator

from dataquality.schemas.metrics import FilterParams


@unique
class EditAction(str, Enum):
    """The available actions you can take in an edit"""

    relabel = "relabel"
    delete = "delete"
    select_for_label = "select_for_label"
    relabel_as_pred = "relabel_as_pred"
    update_text = "update_text"
    shift_span = "shift_span"


class Edit(BaseModel):
    """An extension of the create request where project, run, and split are required"""

    edit_action: EditAction
    filter: Optional[FilterParams]

    new_label: Optional[StrictStr]  # needed if action==relabel

    search_string: Optional[StrictStr]  # needed if action==update_text
    text_replacement: Optional[StrictStr]  # needed if action==update_text
    use_regex: bool = False  # relates to the search_string

    shift_span_start_num_words: Optional[StrictInt]  # Num words to `shift_span` by
    shift_span_end_num_words: Optional[StrictInt]  # Num words to `shift_span` by

    project_id: Optional[UUID4]
    run_id: Optional[UUID4]
    split: Optional[str]
    task: Optional[str] = None
    inference_name: Optional[str] = None
    note: Optional[StrictStr]

    @validator("edit_action", pre=True)
    def new_label_if_relabel(cls, edit_action: EditAction, values: Dict) -> EditAction:
        if edit_action == EditAction.relabel and values["new_label"] is None:
            raise ValueError("If your edit is relabel, you must set new_label")
        return edit_action

    @validator("edit_action", pre=True)
    def text_replacement_if_update_text(
        cls, edit_action: EditAction, values: Dict
    ) -> EditAction:
        if edit_action == EditAction.update_text and (
            values["text_replacement"] is None or values["search_string"] is None
        ):
            raise ValueError(
                "If your edit is update_text, you must set "
                "text_replacement and search_string"
            )
        return edit_action

    @validator("edit_action", pre=True)
    def shift_span_validator(cls, edit_action: EditAction, values: Dict) -> EditAction:
        err = (
            "If your edit is shift_span, you must set text_replacement and at least "
            "one of shift_span_start_num_words or shift_span_end_num_words"
        )
        if edit_action == EditAction.shift_span:
            if values["text_replacement"] is None:
                raise ValueError(err)
            if (
                values["shift_span_start_num_words"] is None
                and values["shift_span_end_num_words"] is None
            ):
                raise ValueError(err)
        return edit_action

    @validator("edit_action", pre=True, always=True)
    def validate_edit_action_for_split(
        cls, edit_action: EditAction, values: Dict[str, Any]
    ) -> EditAction:
        if not values.get("split"):
            return edit_action
        split = values["split"]
        if split == "inference":
            if edit_action != EditAction.select_for_label:
                raise ValueError(f"Invalid edit action {edit_action} for split {split}")
        elif edit_action == EditAction.select_for_label:
            raise ValueError(f"Invalid edit action {edit_action} for split {split}")
        return edit_action
