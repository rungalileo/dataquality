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
    process_batch: bool = True
    # Sample level chat cols
    turns_col: str = "turns"
    metadata_col: str = "metadata"
    # Turn level chat cols
    content_col: str = "content"
    role_col: str = "role"
    # Chat roles
    user: str = "User"
    assistant: str = "Chatbot"

    def format_sample(
        self, sample: Dict[str, Any], idx: Optional[int] = None
    ) -> Dict[str, Any]:
        """Formats a chat dataset for seq2seq

        Takes in a sample with "turns" column and explodes it to have one row
        per turn.

        Example:
            >>> sample = {
            ...     "turns": [
            ...         {"role": "User", "content": "Hello"},
            ...         {"role": "Chatbot", "content": "Hi"},
            ...         {"role": "User", "content": "How are you?"},
            ...         {"role": "Chatbot", "content": "I'm good, how are you?"},
            ...     ],
            ...     "metadata": {"unique_id": 1234, "dataset": "test"},
            ...     "score": 0.5,
            ... }
            >>> ChatFormatter().format_sample(sample, 5)
            {
                "chat_id": [5, 5],
                "turn_id": [1, 2],
                "input": ["Hello", "How are you?"],
                "target": ["Hi", "I'm good, how are you?"],
                "unique_id": [1234, 1234],
                "dataset": ["test", "test"],
            }
        """
        unraveled_turns: Dict[str, Any] = defaultdict(list)
        valid_meta_types = (str, int, float, bool)
        turns: List[Dict[str, Any]] = sample[self.turns_col]

        # # Add metadata and sample level cols to each turn
        metadata: Dict[str, Any] = sample.get(self.metadata_col, {})
        for k, v in sample.items():
            if k not in [self.metadata_col, self.turns_col, "id"]:
                metadata[k] = v

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
            # Add turn level metadata to turn
            # NOTE: When we drop p3.8 we can use 'turn_data |= turn_meta'
            turn_data.update(turn_meta)

            if role == self.user:
                turn_data[self.input_col] = content
            elif role == self.assistant:
                turn_data[self.target_col] = content
                turn_data["turn_id"] = turn_id
                turn_data["chat_id"] = idx
                # Add sample level metadata
                # NOTE: When we drop p3.8 we can use 'turn_data |= turn_meta'
                turn_data.update(metadata)
                for k, v in turn_data.items():
                    unraveled_turns[k].append(v)
                # Reset turn data
                turn_data = {}
                turn_id += 1
            else:
                raise ValueError(f"Role {role} not recognized")

        return unraveled_turns
