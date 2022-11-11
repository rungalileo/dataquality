import dataclasses
from dataclasses import dataclass
from typing import List


@dataclass
class HFCol:
    input_ids: str = "input_ids"
    text: str = "text"
    id: str = "id"
    ner_tags: str = "ner_tags"
    text_token_indices: str = "text_token_indices"
    tokens: str = "tokens"
    bpe_tokens: str = "bpe_tokens"
    gold_spans: str = "gold_spans"
    labels: str = "labels"
    ner_labels: str = "ner_labels"
    tags: str = "tags"

    @staticmethod
    def get_fields() -> List[str]:
        return [field.name for field in dataclasses.fields(HFCol)]


@dataclass
class SpanKey:
    label: str = "label"
    start: str = "start"
    end: str = "end"
