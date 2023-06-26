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


@dataclass
class AlignedTokenData:
    token_label_offsets: List[List[Tuple[int, int]]]
    token_label_positions: List[List[Set[int]]]
