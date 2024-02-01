from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Type

import numpy as np

from dataquality.loggers.logger_config.seq2seq.seq2seq_base import Seq2SeqLoggerConfig
from dataquality.schemas.seq2seq import Seq2SeqModelType
from dataquality.utils.seq2seq import remove_padding


class BaseSeq2SeqModelFormatter(ABC):
    def __init__(self, logger_config: Seq2SeqLoggerConfig) -> None:
        self.logger_config = logger_config

    @abstractmethod
    def format_sample(
        self,
        sample_id: int,
        sample_output_tokens: np.ndarray,
        split_key: str,
        shift_labels: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Formats sample_output_tokens before extracting token information

        Depending on the model architecture this function:
            - Removes padding tokens from model outputs
            - Restricts to just the response / target tokens

        Note: `shift_labels` is only used for DecoderOnly models. See further details
        in the DecoderOnly definition.

        Returns:
            - formatted_labels: np.ndarray
                Used for extracting token logprob data
            - formatted_sample_output_tokens: np.ndarray
        """
        pass

    def retrieve_sample_labels(
        self, sample_id: int, max_tokens: Optional[int], split_key: str
    ) -> np.ndarray:
        """Retrieve the labels array based on the sample id and truncate at max_tokens

        Labels gives the ground truth / target sample ids for
        each token in the sequence:

        e.g. for sample_id = 8 --> labels = [0, 10, 16, ...]
        """
        labels = self.logger_config.id_to_tokens[split_key][sample_id]
        if max_tokens is not None:
            labels = labels[:max_tokens]
        return np.array(labels)


class EncoderDecoderModelFormatter(BaseSeq2SeqModelFormatter):
    """Seq2Seq model logger for EncoderDecoder models

    Since Encoder-Decoder models output logits just over the target tokens,
    there is very little additional processing - i.e. we primarily leverage
    functionality from Seq2SeqModelLogger.
    """

    def format_sample(
        self,
        sample_id: int,
        sample_output_tokens: np.ndarray,
        split_key: str,
        shift_labels: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Formats sample_output_tokens by removing padding tokens"""
        sample_n_tokens = sample_output_tokens.shape[0]
        sample_labels = self.retrieve_sample_labels(
            sample_id, max_tokens=sample_n_tokens, split_key=split_key
        )
        padding_side = getattr(self.logger_config.tokenizer, "padding_side", "right")
        num_sample_tokens = len(sample_labels)
        sample_tokens = remove_padding(
            sample_output_tokens,
            num_sample_tokens,
            padding_side,
        )

        return sample_labels, sample_tokens


class DecoderOnlyModelFormatter(BaseSeq2SeqModelFormatter):
    """Seq2Seq model logger for EncoderDecoder models

    Since Encoder-Decoder models output logits just over the target tokens,
    there is very little additional processing - i.e. we primarily leverage
    functionality from Seq2SeqModelLogger.
    """

    def _retrieve_num_sample_tokens(
        self, sample_id: int, max_tokens: int, split_key: str
    ) -> Tuple[int, Optional[int]]:
        """Retrieves the number of tokens in the formatted_prompt

        This is used to remove padding tokens before restricting to the response tokens.

        If the num_sample_tokens > max_tokens - meaning the user tokenized their
        data with a smaller max_length, for not we should throw an error!

        What this really means is we need to keep track of how many tokens to strip
        from the response labels. Take this e.g.
            - num_sample_tokens = 135
            - max_tokens = 128
            - num_sample_tokens - max_tokens = 7
            - We need to remove the last 7 tokens of the response labels.
            sample_labels = sample_labels[:-7]

        So to handle this, we should return num_sample_tokens and the difference
        if we have max_tokens!
        """
        num_sample_tokens = self.logger_config.id_to_formatted_prompt_length[split_key][
            sample_id
        ]

        if num_sample_tokens > max_tokens:
            return max_tokens, num_sample_tokens - max_tokens

        return num_sample_tokens, None

    def format_sample(
        self,
        sample_id: int,
        sample_output_tokens: np.ndarray,
        split_key: str,
        shift_labels: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Formats sample_output_tokens

        Actions taken:
            - Removes padding tokens based off of the length of the tokenized
            formatted prompt
            - Restricts to just response tokens using the saved response_labels

        The shift_labels flag is used to align the 'logits' / 'logprobs' with the
        Response Token Labels. As a general rule:
            - When logging directly from non-api models (e.g. hf), the response_labels
            are "shifted" right by one from the logits. Thus, to align them - i.e.
            get the correct logits for each token label - we need to account for this
            shift.
                e.g.
                formatted_sample_ids = [1, 2, 3, 4, 5, 6, 7, 8]
                response_tokens_ids = [6, 7, 8]
                logits = shape[8, vocab]

                # Output corresponsing to model input tokens [5, 6, 7]
                response_logits = logits[-4: -1]
                # NOT
                response_logits = logits[-3:]

            - When logging from an api, the logits or logprobs are generally aligned for
            us. Therefore, we don't need to account for this right shift.
        """
        sample_n_tokens = sample_output_tokens.shape[0]
        num_sample_labels, num_extra_tokens = self._retrieve_num_sample_tokens(
            sample_id, sample_n_tokens, split_key
        )

        response_labels = self.retrieve_sample_labels(
            sample_id, max_tokens=None, split_key=split_key
        )
        # TODO Check this logic - especially around getting the correct
        #   sample size!
        if num_extra_tokens:
            response_labels = response_labels[:-num_extra_tokens]

        padding_side = getattr(self.logger_config.tokenizer, "padding_side", "right")
        sample_wo_padding = remove_padding(
            sample_output_tokens, num_sample_labels, padding_side
        )

        # Restrict to just the response tokens
        num_response_tokens = len(response_labels)
        # Shift sample tokens if necessary such that tokens < n predict token n.
        if shift_labels:
            sample_response = sample_wo_padding[-(num_response_tokens + 1) : -1]
        else:
            sample_response = sample_wo_padding[-num_response_tokens:]

        return response_labels, sample_response


FORMATTER_MAP: Dict[Seq2SeqModelType, Type[BaseSeq2SeqModelFormatter]] = {
    Seq2SeqModelType.encoder_decoder: EncoderDecoderModelFormatter,
    Seq2SeqModelType.decoder_only: DecoderOnlyModelFormatter,
}


def get_model_formatter(
    model_type: Seq2SeqModelType, logger_config: Seq2SeqLoggerConfig
) -> BaseSeq2SeqModelFormatter:
    """Returns the model formatter for the given model_type"""
    return FORMATTER_MAP[model_type](logger_config)
