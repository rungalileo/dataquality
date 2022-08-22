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
    """A class for help creating edits via dq.metrics
    An edit is a combination of a filter, and some edit action. You can use this
    class, as well as `dq.metrics.create_edit` and `dq.metrics.get_edited_dataframe`
    to create automated edits and improved datasets, leading to automated retraining
    pipelines.
    Args:
        edit_action: EditAction the type of edit.
            delete, relabel, relabel_as_pred, update_text, shift_span (ner only), and
            select_for_label (inference only)
        new_label: Optional[str] needed if action is relabel, ignored otherwise. The
            new label to set for the edit
        search_string: Optional[str] needed when action is text replacement or
            shift_span. The search string to use for the edit
        use_regex: bool = False. Used for the search_string. When searching, whether to
            use regex or not. Default False.
        shift_span_start_num_words: Optional[int] Needed if action is shift_span.
            How many words (forward or back) to shift the beginning of the span by
        shift_span_end_num_words: Optional[int] Needed if action is shift_span.
            How many words (forward or back) to shift the end of the span by
    """

    filter: Optional[FilterParams]

    new_label: Optional[StrictStr]

    search_string: Optional[StrictStr]
    text_replacement: Optional[StrictStr]
    use_regex: bool = False

    shift_span_start_num_words: Optional[StrictInt]
    shift_span_end_num_words: Optional[StrictInt]

    project_id: Optional[UUID4]
    run_id: Optional[UUID4]
    split: Optional[str]
    task: Optional[str] = None
    inference_name: Optional[str] = None
    note: Optional[StrictStr]
    edit_action: EditAction

    @validator("edit_action", pre=True)
    def new_label_if_relabel(cls, edit_action: EditAction, values: Dict) -> EditAction:
        if edit_action == EditAction.relabel and values.get("new_label") is None:
            raise ValueError("If your edit is relabel, you must set new_label")
        return edit_action

    @validator("edit_action", pre=True)
    def text_replacement_if_update_text(
        cls, edit_action: EditAction, values: Dict
    ) -> EditAction:
        if edit_action == EditAction.update_text and (
            values.get("text_replacement") is None
            or values.get("search_string") is None
        ):
            raise ValueError(
                "If your edit is update_text, you must set "
                "text_replacement and search_string"
            )
        return edit_action

    @validator("edit_action", pre=True)
    def shift_span_validator(cls, edit_action: EditAction, values: Dict) -> EditAction:
        err = (
            "If your edit is shift_span, you must set search_string and at least "
            "one of shift_span_start_num_words or shift_span_end_num_words"
        )
        if edit_action == EditAction.shift_span:
            if values.get("search_string") is None:
                raise ValueError(err)
            if (
                values.get("shift_span_start_num_words") is None
                and values.get("shift_span_end_num_words") is None
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
