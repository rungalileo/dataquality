from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Type


@dataclass
class BaseFormatter(ABC):
    name: str
    input_col: str
    target_col: str

    @abstractmethod
    def format_sample(self, sample: Dict[str, str]) -> Dict[str, str]:
        """Base formatter is identity function"""
        pass


@dataclass
class DefaultFormatter(BaseFormatter):
    name: str = "default"
    input_col: str = "text"
    target_col: str = "label"

    def format_sample(self, sample: Dict[str, str]) -> Dict[str, str]:
        """Base formatter is identity function"""
        return sample


@dataclass
class AlpacaFormatter(BaseFormatter):
    name: str = "tatsu-lab/alpaca"
    input_col: str = "formatted_input"
    target_col: str = "output"

    def format_sample(self, sample: Dict[str, str]) -> Dict[str, str]:
        """Formats the alpaca dataset for seq2seq

        Example:
            >>> sample = {
            ...     "instruction": "Summarize the following paragraph",
            ...     "input": "The quick brown fox jumped over the lazy dog.",
            ...     "target": "The quick brown fox jumped over the lazy dog.",
            ... }
            >>> format_alpaca(sample)
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


FORMATTER_MAPPING: Dict[str, Type[BaseFormatter]] = {
    AlpacaFormatter.name: AlpacaFormatter,
}


def get_formatter(name: str) -> BaseFormatter:
    """Returns the formatter for the given name

    If the name isn't found, returns the base formatter
    """
    return FORMATTER_MAPPING.get(name, DefaultFormatter)()  # type: ignore
