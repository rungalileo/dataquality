from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class BatchData:
    batch: Dict[str, Any]

    def sample_from_idx(self, batch_idx: int) -> Dict[str, Any]:
        """Gets a subset of the batch"""
        sample = {}
        for k, v in self.batch.items():
            sample[k] = v[batch_idx]
        return sample


@dataclass
class BaseFormatter(ABC):
    name: str
    input_col: str
    target_col: str
    max_train_size: Optional[int] = None
    process_batch: bool = False

    @property
    def remove_cols(self) -> List[str]:
        return []

    def format_batch(self, batch: Dict, idxs: List[int]) -> Dict[str, List]:
        """Formats a batch of chat data for seq2seq"""
        result: Dict[str, List] = defaultdict(list)
        batch_data = BatchData(batch)
        batch_sz = len(idxs)
        for idx in idxs:
            batch_idx = idx % batch_sz
            formatted_sample = self.format_sample(
                batch_data.sample_from_idx(batch_idx), idx
            )
            # formatted_sample returns one or more samples per idx, we add to result
            for k, v in formatted_sample.items():
                result[k] += v

        return result

    @abstractmethod
    def format_sample(
        self, sample: Dict[str, Any], idx: Optional[int] = None
    ) -> Dict[str, Any]:
        """Must be implemented by subclass"""
        pass


@dataclass
class DefaultFormatter(BaseFormatter):
    name: str = "default"
    input_col: str = "input"
    target_col: str = "target"

    def format_sample(
        self, sample: Dict[str, Any], idx: Optional[int] = None
    ) -> Dict[str, Any]:
        """Base formatter is identity function"""
        return sample
