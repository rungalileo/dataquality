from enum import Enum, unique


@unique
class NERErrorType(str, Enum):
    wrong_tag = "wrong_tag"
    missed_label = "missed_label"
    span_shift = "span_shift"
    none = "None"
