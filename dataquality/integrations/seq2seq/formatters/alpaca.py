from dataclasses import dataclass
from typing import Dict, List, Optional

from dataquality.integrations.seq2seq.formatters.base import BaseFormatter


@dataclass
class AlpacaFormatter(BaseFormatter):
    name: str = "tatsu-lab/alpaca"
    input_col: str = "formatted_input"
    target_col: str = "output"
    max_train_size: int = 1000

    @property
    def remove_cols(self) -> List[str]:
        return ["input", "text"]

    def format_sample(
        self, sample: Dict[str, str], idx: Optional[int] = None
    ) -> Dict[str, str]:
        """Formats the alpaca dataset for seq2seq

        Example:
            >>> sample = {
            ...     "instruction": "Summarize the following paragraph",
            ...     "input": "The quick brown fox jumped over the lazy dog.",
            ...     "target": "The quick brown fox jumped over the lazy dog.",
            ... }
            >>> AlpacaFormatter().format_sample(sample)
            {
                "formatted_input": (
                    "Human: Summarize the following paragraph "
                    "Context: The quick brown fox jumped over the lazy dog."
                )
            }
        """
        instruction = f"Human: {sample['instruction']}"
        # By multiplying by a bool, we only add the context if it exists
        context = f"Context: {sample['input']}" * bool(sample["input"])
        return {
            "formatted_input": f"{instruction} {context}",
        }
