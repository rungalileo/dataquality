from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pyarrow as pa
from scipy.special import log_softmax

from dataquality.exceptions import GalileoException
from dataquality.loggers.logger_config.seq2seq.seq2seq_base import (
    Seq2SeqLoggerConfig,
    seq2seq_logger_config,
)
from dataquality.loggers.model_logger.base_model_logger import BaseGalileoModelLogger
from dataquality.loggers.model_logger.seq2seq.formatters import (
    BaseSeq2SeqModelFormatter,
    get_model_formatter,
)
from dataquality.schemas.seq2seq import TOP_K, TOP_LOGPROBS_SCHEMA
from dataquality.schemas.seq2seq import Seq2SeqOutputCols as C
from dataquality.schemas.split import Split
from dataquality.utils.arrow import save_arrow_file
from dataquality.utils.emb import np_to_pa
from dataquality.utils.helpers import has_len
from dataquality.utils.seq2seq.logprobs import (
    get_top_logprob_indices,
    process_sample_logprobs,
)


class Seq2SeqModelLogger(BaseGalileoModelLogger):
    """Seq2Seq base model logger

    This class defines the base functionality for logging model outputs in
    Seq2Seq tasks - shared between EncoderDecoder and DecoderOnly architectures.

    After architecture specific processing of raw model logits, we leverage
    a shared function for processing and extracting the logprob token data **just**
    over the Target data.

    During processing, the following key information is extracted:
        - token_logprobs: log-probs for GT tokens in each sample
        - top_logprobs: top-K (token_str, log-prob) pairs for each token
    """

    __logger_name__ = "seq2seq"
    logger_config: Seq2SeqLoggerConfig = seq2seq_logger_config
    log_file_ext = "arrow"

    def __init__(
        self,
        embs: Optional[Union[List, np.ndarray]] = None,
        probs: Optional[Union[List, np.ndarray]] = None,
        logits: Optional[Union[List, np.ndarray]] = None,
        ids: Optional[Union[List, np.ndarray]] = None,
        split: str = "",
        epoch: Optional[int] = None,
        inference_name: Optional[str] = None,
    ) -> None:
        """Initialize the Seq2SeqModelLogger

        In Seq2Seq if probs is passed in it is actually logprobs
        """
        super().__init__(
            embs=embs,
            probs=probs,
            logits=logits,
            ids=ids,
            split=split,
            epoch=epoch,
            inference_name=inference_name,
        )
        self.token_logprobs = pa.array([])
        self.top_logprobs = pa.array([])
        # Formatter distinguishes behavior between EncoderDecoder and DecoderOnly
        self.formatter: Optional[BaseSeq2SeqModelFormatter] = None

    @property
    def split_key(self) -> str:
        if self.split == Split.inference and self.inference_name is not None:
            return self.inference_name
        return str(self.split)

    def validate_and_format(self) -> None:
        """Validate the lengths, calculate token level dep, extract GT probs"""
        self.embs = self._convert_tensor_ndarray(self.embs)
        self.logits = self._convert_tensor_ndarray(self.logits)
        # Note that for seq2seq if probs is set they are actually logprobs
        self.probs = self._convert_tensor_ndarray(self.probs)
        self.ids = self._convert_tensor_ndarray(self.ids)
        assert (len(self.ids) == len(self.logits)) or (
            len(self.ids) == len(self.probs)
        ), (
            "Must pass in a valid batch with equal id and logit/probs length, got "
            f"id: {len(self.ids)},logits: {len(self.logits)},probs: {len(self.probs)}"
        )

        assert self.logger_config.tokenizer is not None, (
            "Must set your tokenizer. Use `dq.integrations.seq2seq.core.watch` or "
            "`dq.integrations.seq2seq.core.watch`"
        )

        model_type = self.logger_config.model_type
        if model_type is None:
            raise GalileoException(
                "You must set your model type before logging. Use "
                "`dataquality.integrations.seq2seq.core.watch`"
            )

        # Now that model_type has been set with `watch` we set formatter
        self.formatter = get_model_formatter(model_type, self.logger_config)

        if has_len(self.probs):
            self.token_logprobs, self.top_logprobs = self.process_logprobs(
                self.ids, self.probs
            )
        else:
            (
                self.token_logprobs,
                self.top_logprobs,
            ) = self.process_logits(
                self.ids, self.logits  # type: ignore
            )

    def process_logits(
        self, batch_ids: np.ndarray, batch_logits: np.ndarray
    ) -> Tuple[pa.array, pa.array]:
        """Process a batch of sample logit data

        For each sample in the batch extract / compute the following values:
            - Token level logprobs for the GT label
            - Token level top-k model logprobs: represented as a dictionary
            mapping {predicted_token: logprob}

        batch_logits has shape - [batch_size, max_token_length, vocab_size], where
        max_token_length is determined by the longest sample in the batch. Because
        other samples in the batch are padded to this max_length, we have to process
        each sample individually to ignore pad token indices.

        Special points of consideration:
            - For each sample, top-k logprobs is a list of dictionaries with length
            equal to the number of tokens in that sample. Each dictionary maps the
            models top-k predicted tokens to their corresponding logprobs.

            - We return a pyarrow array because each sample may have a different number
            of token, which can't be represented in numpy.

        Returns:
            batch_token_logprobs: GT Logprob per token
                len(batch_token_dep) == batch_size
                batch_token_logprobs[i].shape is [num_tokens_in_label[i]]
            batch_top_logprobs: Top-k logprob dictionary per token
                type(batch_top_logprobs[i]) = List[Dict[str, float]]
                len(batch_top_logprobs) == batch_size
                len(batch_top_logprobs[i]) = num_tokens_in_label[i]
        """
        # Formatter and tokenizer will have already been set in `validate_and_format`
        # These are needed for linting
        assert self.logger_config.tokenizer is not None
        assert self.formatter is not None

        batch_token_logprobs = []
        batch_top_logprobs = []
        # Iterate through the samples in the batch
        for sample_id, sample_logits in zip(batch_ids, batch_logits):
            (
                sample_labels,
                sample_logits,
            ) = self.formatter.format_sample(sample_id, sample_logits, self.split_key)

            sample_logprobs = self.convert_logits_to_logprobs(sample_logits)
            sample_top_indices = get_top_logprob_indices(sample_logprobs)

            logprob_data = process_sample_logprobs(
                sample_logprobs=sample_logprobs,
                sample_labels=sample_labels,
                sample_top_indices=sample_top_indices,
                tokenizer=self.logger_config.tokenizer,
            )
            batch_token_logprobs.append(logprob_data.token_logprobs)
            batch_top_logprobs.append(logprob_data.top_logprobs)

        return (
            pa.array(batch_token_logprobs),
            pa.array(batch_top_logprobs, type=TOP_LOGPROBS_SCHEMA),
        )

    def process_logprobs(
        self, batch_ids: np.ndarray, batch_logprobs: np.ndarray
    ) -> Tuple[pa.array, pa.array]:
        """Process a batch of sample logprob data

        This is a special case where the use only logs a single logprobs
        for each token - i.e. the label token's logprob.

            batch_logprobs.shape = [bs, max_token_length]

        In this case, we do not have any `top_k` logprob data; therefore,
        we fill the top_logprob data with "filler" data. Each token's
        top 5 logprob data is:

            [("---", -20)] * TOP_K

        Similar to `process_logits` we process the logprob data to remove
        1) remove padding and 2) apply any other formatting to just restrict
        to token level information for the "Target" tokens.


        Special points of consideration:
            - We return a pyarrow array because each sample may have a different number
            of token, which can't be represented in numpy.

        Returns:
            batch_token_logprobs: GT Logprob per token
                len(batch_token_dep) == batch_size
                batch_token_logprobs[i].shape is [num_tokens_in_label[i]]
            batch_top_logprobs: Top-k logprob dictionary per token
                type(batch_top_logprobs[i]) = List[Dict[str, float]]
                len(batch_top_logprobs) == batch_size
                len(batch_top_logprobs[i]) = num_tokens_in_label[i]
                batch_top_logprobs[i][0] = ("---", -20)
        """
        # Formatter will have already been set in `validate_and_format`
        # These are needed for linting
        assert self.formatter is not None

        batch_token_logprobs = []
        batch_top_logprobs = []
        for sample_id, sample_logprobs in zip(batch_ids, batch_logprobs):
            # API based models will have already shifted the logprobs
            sample_labels, sample_response_logprobs = self.formatter.format_sample(
                sample_id, sample_logprobs, self.split_key, shift_labels=False
            )

            # Add fake top loprobs
            sample_top_logprobs: List[List[Tuple[str, float]]] = [
                [("---", -20)] * TOP_K
            ] * len(sample_labels)

            batch_token_logprobs.append(sample_response_logprobs)
            batch_top_logprobs.append(sample_top_logprobs)

        return (
            pa.array(batch_token_logprobs),
            pa.array(batch_top_logprobs, type=TOP_LOGPROBS_SCHEMA),
        )

    def _get_data_dict(self) -> Dict:
        """Returns the data dictionary for writing to disk"""
        batch_size = len(self.ids)
        data = {
            C.id.value: self.ids,
            C.token_logprobs.value: self.token_logprobs,
            C.top_logprobs.value: self.top_logprobs,
            C.split_.value: [Split[self.split].value] * batch_size,
            C.epoch.value: [self.epoch] * batch_size,
        }
        if self.embs is not None:
            # In seq2seq we have to save embs as a pyarrow array instead of numpy
            # since the vaex DataFrames are stored as arrow files
            if not isinstance(self.embs, np.ndarray):
                self.embs = self._convert_tensor_ndarray(self.embs)
            if self.embs.shape[0] > 0:
                data[C.emb.value] = np_to_pa(self.embs)

        if self.split == Split.inference:
            data[C.inference_name.value] = [self.inference_name] * batch_size
        return data

    def _write_dict_to_disk(self, path: str, object_name: str, data: Dict) -> None:
        save_arrow_file(path, object_name, data)

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
            sample_logits = self._convert_tensor_ndarray(sample_logits)

        return log_softmax(sample_logits, axis=-1)
