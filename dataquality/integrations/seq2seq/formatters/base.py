from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class BaseFormatter(ABC):
    name: str
    input_col: str
    target_col: str
    max_train_size: Optional[int] = None
    remove_columns: bool = False

    def format_batch(self, batch: Dict, idxs: List[int]) -> Dict[str, List]:
        """Formats a batch of chat data for seq2seq"""
        result: Dict[str, List] = defaultdict(list)
        sample = {}
        for idx in idxs:
            for k, v in batch.items():
                sample[k] = v[idx]
            formatted_sample = self.format_sample(sample, idx)
            for k, v in formatted_sample.items():
                result[k] += v

        return result

    @abstractmethod
    def format_sample(self, sample: Dict[str, Any], idx: int) -> Dict[str, Any]:
        """Base formatter is identity function"""
        pass


@dataclass
class DefaultFormatter(BaseFormatter):
    name: str = "default"
    input_col: str = "text"
    target_col: str = "label"

    def format_sample(self, sample: Dict[str, Any], idx: int) -> Dict[str, Any]:
        """Base formatter is identity function"""
        sample["id"] = idx
        return sample
