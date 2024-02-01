from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from transformers import AutoTokenizer, PreTrainedTokenizerFast

from dataquality.integrations.seq2seq.formatters.base import BaseFormatter
from dataquality.schemas.seq2seq import Seq2SeqInputCols as S2SIC

# HF tokenizers don't support newlines, so we use a token to represent them
# Example of tokenizer without the NEWLINE token:
# result = tokenizer.decode(tokenizer("Hello\n\n\nWorld"))
# print(result)
# >>> HelloWorld
NEWLINE_CHAR = "\n\n\n"
NEWLINE_TOKEN = "[NEWLINE]"


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
    system: str = "System"

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

        turn_data: Dict[str, Any] = {"chat_id": None, "turn_id": None}
        turn_id = 1
        turn_default_cols = [self.role_col, self.content_col]
        system_prompts: Dict[str, str] = {}
        for turn in turns:
            role = turn[self.role_col]
            content = turn[self.content_col]
            if role == self.system:
                system_prompts[self.system] = content
                continue

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
                turn_data[S2SIC.system_prompts] = system_prompts
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


@dataclass
class ChatHistoryFormatter(ChatFormatter):
    hf_tokenizer: Optional[str] = "google/flan-t5-base"
    tokenizer: Optional[PreTrainedTokenizerFast] = None
    max_input_tokens: int = 512

    def __post_init__(self) -> None:
        if self.hf_tokenizer and not self.tokenizer:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.hf_tokenizer,
                truncation_side="left",
                use_fast=True,
                model_max_length=self.max_input_tokens,
            )
        if self.tokenizer:
            self.tokenizer.add_tokens([NEWLINE_TOKEN])

    def format_sample(
        self, sample: Dict[str, Any], idx: Optional[int] = None
    ) -> Dict[str, Any]:
        """Formats a chat dataset for seq2seq with previous turn history

        Similar to ChatFormatter, except subsequent turns contain
        context from previous turns.

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
            >>> ChatHistoryFormatter().format_sample(sample, 5)
            {
                "chat_id": [5, 5],
                "turn_id": [1, 2],
                "input": ["Hello", "Hello\n\nHi\n\nHow are you?"],
                "target": ["Hi", "I'm good, how are you?"],
                "unique_id": [1234, 1234],
                "dataset": ["test", "test"],
            }
        """
        if not self.tokenizer:
            raise ValueError(
                "ChatHistoryFormatter requires a tokenizer to format the input"
            )

        formatted_sample = super().format_sample(sample, idx)
        # Next we update formatted_sample in place one turn at a time
        # We build iteratively from the first turn so that we can add just the previous
        # turn to the input, and it will contain content from all previous turns
        user_inputs = formatted_sample[self.input_col]
        assistant_targets = formatted_sample[self.target_col]
        for i in range(len(user_inputs)):
            # We build up the user input with previous turn history
            history_input = ""
            if i > 0:
                # We add the previous turn input to the history
                history_input += f"{user_inputs[i - 1]}{NEWLINE_CHAR}"
                # We also add the previous turn target (assistant response)
                history_input += (
                    f"{self.assistant}: {assistant_targets[i - 1]}{NEWLINE_CHAR}"
                )

            # Finally we add the current turn input
            history_input += f"{self.user}: {user_inputs[i]}"
            history_input = history_input.replace(NEWLINE_CHAR, NEWLINE_TOKEN)
            # Only take the last max_input_tokens
            history_tokens = self.tokenizer(history_input, truncation=True)["input_ids"]
            parsed_history = self.tokenizer.decode(
                history_tokens, skip_special_tokens=True
            )
            parsed_history = parsed_history.replace(NEWLINE_TOKEN, NEWLINE_CHAR)
            # We don't want to have the input start mid turn, so we find the first
            # instance of the user or assistant role and start there
            first_user_index = parsed_history.find(f"{self.user}: ")
            first_assistant_index = parsed_history.find(f"{self.assistant}: ")
            # .find() returns -1 if the substring is not found, we must ignore those
            non_negative = [
                val for val in [first_user_index, first_assistant_index] if val >= 0
            ]
            # If both are -1, we just take the last max_input_tokens tokens
            start_index = min(non_negative) if non_negative else -self.max_input_tokens
            user_inputs[i] = f"{parsed_history[start_index:]}"

        return formatted_sample
