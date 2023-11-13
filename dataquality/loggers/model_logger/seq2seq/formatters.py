from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Type, Union

import numpy as np
from scipy.special import log_softmax

from dataquality.loggers.base_logger import BaseGalileoLogger
from dataquality.loggers.logger_config.seq2seq.seq2seq_base import Seq2SeqLoggerConfig
from dataquality.schemas.seq2seq import Seq2SeqModelTypes
from dataquality.utils.seq2seq import remove_padding
from dataquality.utils.seq2seq.logprobs import (
    get_top_logprob_indices,
)


class BaseSeq2SeqModelFormatter(ABC):
    def __init__(self, logger_config: Seq2SeqLoggerConfig, split_key: str) -> None:
        self.logger_config = logger_config
        self.split_key = split_key

    @abstractmethod
    def format_sample(
        self, sample_id: int, sample_logits: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pass

    def retrieve_sample_labels(
        self, sample_id: int, max_tokens: Optional[int]
    ) -> np.ndarray:
        """Retrieve the labels array based on the sample id and truncate at max_tokens

        Labels gives the ground truth / target sample ids for
        each token in the sequence:

        e.g. for sample_id = 8 --> labels = [0, 10, 16, ...]
        """
        labels = self.logger_config.id_to_tokens[self.split_key][sample_id]
        if max_tokens is not None:
            labels = labels[:max_tokens]
        return np.array(labels)

    def convert_logits_to_logprobs(
        self, sample_logits: Union[List[np.ndarray], np.ndarray]
    ) -> np.ndarray:
        """Converts logits (unnormalized log probabilities) to logprobs via log_softmax

        This is a special use case for Seq2Seq, people generally
        work with logprobs. One reason for this is that the logsoftmax
        function takes advantage of the logsumexp "trick" to compute a
        numerically stable version of log(softmax(x)).
        """
        # axis ensures that in a matrix of probs with dims num_samples x num_classes
        # we take the softmax for each sample
        if not isinstance(sample_logits, np.ndarray):
            sample_logits = BaseGalileoLogger._convert_tensor_ndarray(sample_logits)

        return log_softmax(sample_logits, axis=-1)


class EncoderDecoderModelFormatter(BaseSeq2SeqModelFormatter):
    """Seq2Seq model logger for EncoderDecoder models

    Since Encoder-Decoder models output logits just over the target tokens,
    there is very little additional processing - i.e. we primarily leverage
    functionality from Seq2SeqModelLogger.
    """

    def format_sample(
        self, sample_id: int, sample_logits: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Formats sample_logprobs and sample_top_indices

        Removes padding.

        Returns:
            - formatted_labels: np.ndarray
            - formatted_sample_logprobs: np.ndarray
            - formatted_sample_top_indices: np.ndarray
        """
        sample_n_tokens = sample_logits.shape[0]
        # TODO this could be abstracted away
        sample_labels = self.retrieve_sample_labels(
            sample_id, max_tokens=sample_n_tokens
        )
        padding_side = getattr(self.logger_config.tokenizer, "padding_side", "right")
        num_sample_tokens = len(sample_labels)
        sample_logits = remove_padding(
            sample_logits,
            num_sample_tokens,
            padding_side,
        )

        sample_logprobs = self.convert_logits_to_logprobs(sample_logits)
        sample_top_indices = get_top_logprob_indices(sample_logprobs)

        return sample_labels, sample_logprobs, sample_top_indices


class DecoderOnlyModelFormatter(BaseSeq2SeqModelFormatter):
    """Seq2Seq model logger for EncoderDecoder models

    Since Encoder-Decoder models output logits just over the target tokens,
    there is very little additional processing - i.e. we primarily leverage
    functionality from Seq2SeqModelLogger.
    """

    def _retrieve_num_sample_tokens(
        self, sample_id: int, max_tokens: int
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
        num_sample_tokens = self.logger_config.id_to_formatted_prompt_length[
            self.split_key
        ][sample_id]

        if num_sample_tokens > max_tokens:
            return max_tokens, num_sample_tokens - max_tokens

        return num_sample_tokens, None

    def format_sample(
        self, sample_id: int, sample_logits: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Formats sample_logprobs and sample_top_indices

        Removes padding and (for DecoderOnly models) restricts to just
        response tokens.

        Returns:
            - formatted_labels: np.ndarray
            - formatted_sample_logprobs: np.ndarray
            - formatted_sample_top_indices: np.ndarray
        """
        sample_n_tokens = sample_logits.shape[0]
        num_sample_labels, num_extra_tokens = self._retrieve_num_sample_tokens(
            sample_id, sample_n_tokens
        )

        response_labels = self.retrieve_sample_labels(sample_id, max_tokens=None)
        # TODO Check this logic - especially around getting the correct
        #   sample size!
        if num_extra_tokens:
            response_labels = response_labels[:-num_extra_tokens]

        padding_side = getattr(self.logger_config.tokenizer, "padding_side", "right")
        sample_logits = remove_padding(sample_logits, num_sample_labels, padding_side)

        # Restrict to just the response tokens
        num_response_tokens = len(response_labels)
        # TODO check - Shift the logits such that tokens < n predict token n.
        #   notice here that we ignore the final token logprob since there is
        #   no n+1 token. For DecoderOnly the logits and labels are implicitly
        #   shifted within the model.
        sample_logits = sample_logits[-(num_response_tokens + 1) : -1]

        sample_logprobs = self.convert_logits_to_logprobs(sample_logits)
        sample_top_indices = get_top_logprob_indices(sample_logprobs)

        return response_labels, sample_logprobs, sample_top_indices


FORMATTER_MAP: Dict[Seq2SeqModelTypes, Type[BaseSeq2SeqModelFormatter]] = {
    Seq2SeqModelTypes.encoder_decoder: EncoderDecoderModelFormatter,
    Seq2SeqModelTypes.decoder_only: DecoderOnlyModelFormatter,
}


def get_model_formatter(
    model_type: Seq2SeqModelTypes, logger_config: Seq2SeqLoggerConfig, split_key: str
) -> BaseSeq2SeqModelFormatter:
    """Returns the model formatter for the given model_type"""
    return FORMATTER_MAP[model_type](logger_config, split_key)
