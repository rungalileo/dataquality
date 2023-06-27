from dataclasses import dataclass
from enum import Enum
from typing import List, Set, Tuple


class Seq2SeqCols(str, Enum):
    text = "text"
    id = "id"
    label = "label"
    tokenized_label = "tokenized_label"
    token_label_positions = "token_label_positions"
    token_label_offsets = "token_label_offsets"
    token_deps = "token_deps"
    token_gold_probs = "token_gold_probs"
    # Mypy complained about split as an attribute, so we use `split_`
    split_ = "split"
    epoch = "epoch"


@dataclass
class AlignedTokenData:
    token_label_offsets: List[List[Tuple[int, int]]]
    token_label_positions: List[List[Set[int]]]
