from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from dataquality.integrations.seq2seq.formatters.base import BaseFormatter


@dataclass
class ChatFormatter(BaseFormatter):
    name: str = "chat"
    input_col: str = "input"
    target_col: str = "target"
    max_train_size: Optional[int] = None
    remove_columns: bool = True
    # Chat specific cols
    content_col: str = "content"
    turns_col: str = "turns"
    role_col: str = "role"
    role_1: str = "User"
    role_2: str = "Chatbot"
    metadata_col: str = "metadata_col"

    def format_sample(self, sample: Dict[str, Any], idx: int) -> Dict[str, Any]:
        """Formats a chat dataset for seq2seq

        Takes in a sample with "turns" column and explodes it to have one row
        per turn.

        Example:
            >>> sample = {
            ...     "instruction": "Summarize the following paragraph",
            ...     "input": "The quick brown fox jumped over the lazy dog.",
            ...     "target": "The quick brown fox jumped over the lazy dog.",
            ... }
            >>> ChatFormatter().format_sample(sample)
            {
                "formatted_input": (
                    "Human: Summarize the following paragraph "
                    "Context: The quick brown fox jumped over the lazy dog."
                )
            }
        """
        unraveled_turns: Dict[str, Any] = defaultdict(list)
        valid_meta_types = (str, int, float, bool)
        turns: List[Dict[str, Any]] = sample[self.turns_col]

        # # Add metadata and sample level cols to each turn
        metadata = sample.get(self.metadata_col, {})
        sample_cols = [
            col
            for col in sample.keys()
            if col not in [self.metadata_col, self.turns_col]
        ]
        for col in sample_cols:
            metadata[col] = sample[col]
        unraveled_turns = unraveled_turns | metadata

        turn_data: Dict[str, Any] = {}
        turn_id = 1
        turn_default_cols = [self.role_col, self.content_col]
        for turn in turns:
            role = turn[self.role_col]
            content = turn[self.content_col]
            # Add metadata to each turn
            turn_meta = {
                f"{role}_{col}": turn[col]
                for col in turn.keys()
                if col not in turn_default_cols
                and isinstance(turn[col], valid_meta_types)
            }
            turn_data = turn_data | turn_meta

            if role == self.role_1:
                turn_data[self.input_col] = content
            elif role == self.role_2:
                turn_data[self.target_col] = content
                turn_data["turn_id"] = turn_id
                turn_data["chat_id"] = idx
                for k, v in turn_data.items():
                    unraveled_turns[k].append(v)
                # Reset turn data
                turn_data = {}
                turn_id += 1
            else:
                raise ValueError(f"Role {role} not recognized")

        return unraveled_turns
