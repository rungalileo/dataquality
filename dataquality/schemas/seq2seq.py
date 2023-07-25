from dataclasses import dataclass
from enum import Enum
from typing import List, Set, Tuple


class Seq2SeqInputCols(str, Enum):
    id = "id"
    text = "text"
    label = "label"
    tokenized_label = "tokenized_label"
    token_label_positions = "token_label_positions"
    token_label_offsets = "token_label_offsets"


class Seq2SeqOutputCols(str, Enum):
    id = "id"
    dep = "data_error_potential"
    token_deps = "token_deps"
    token_gold_probs = "token_gold_probs"
    # Mypy complained about split as an attribute, so we use `split_`
    split_ = "split"
    epoch = "epoch"
    inference_name = "inference_name"


@dataclass
class AlignedTokenData:
    token_label_offsets: List[List[Tuple[int, int]]]
    token_label_positions: List[List[Set[int]]]
