from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pyarrow as pa
from scipy.special import log_softmax

from dataquality.loggers.logger_config.seq2seq import (
    Seq2SeqLoggerConfig,
    seq2seq_logger_config,
)
from dataquality.loggers.model_logger.base_model_logger import BaseGalileoModelLogger
from dataquality.schemas.seq2seq import TOP_LOGPROBS_SCHEMA
from dataquality.schemas.seq2seq import Seq2SeqOutputCols as C
from dataquality.schemas.split import Split
from dataquality.utils.arrow import save_arrow_file
from dataquality.utils.seq2seq import get_top_logprob_indices, process_sample_logprobs


class Seq2SeqModelLogger(BaseGalileoModelLogger):
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
        labels: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__(
            embs=embs,
            probs=probs,
            logits=logits,
            ids=ids,
            split=split,
            epoch=epoch,
            inference_name=inference_name,
            labels=labels,
        )
        # TODO Move DEP and Perplexity computation outside of DQ
        self.sample_dep: List[float] = []
        self.sample_perplexity: List[float] = []
        self.token_deps = pa.array([])
        self.token_logprobs = pa.array([])
        self.top_logprobs = pa.array([])
        self.labels = labels

    @property
    def token_map_key(self) -> str:
        if self.split == Split.inference and self.inference_name is not None:
            return self.inference_name
        return str(self.split)

    def validate_and_format(self) -> None:
        """Validate the lengths, calculate token level dep, extract GT probs"""
        if self.labels is not None:
            self.labels = self._convert_tensor_ndarray(self.labels)
        self.logits = self._convert_tensor_ndarray(self.logits)
        self.ids = self._convert_tensor_ndarray(self.ids)
        assert len(self.ids) == len(self.logits), (
            "Must pass in a valid batch with equal id and logit length, got "
            f"id: {len(self.ids)},logits: {len(self.logits)}"
        )
        if self.labels is not None:
            assert len(self.labels) == len(self.ids), "TODO: Must be same len message"

        assert (
            self.logger_config.tokenizer is not None
        ), "Must set your tokenizer. Use `dq.set_tokenizer`"

        # TODO: This is potentially slow. This is what needs to be optimized. Can we
        #  potentially do this on the GPU with torch? And dont convert to a np array
        #  [JON] computing softmax on GPU can lead to speedups of around 5x in my
        #  experience
        logprobs = self.convert_logits_to_logprobs(self.logits)
        (
            self.token_logprobs,
            self.top_logprobs,
            self.token_deps,
            self.sample_dep,
            self.sample_perplexity,
        ) = self.process_logprobs(self.ids, logprobs)

    def compute_token_deps(
        self, sample_logprobs: np.ndarray, labels: np.ndarray
    ) -> np.ndarray:
        """Compute per token DEP

        Per token DEP is computed exactly as in classification. For
        each token we extract the probability for the ground truth token
        and for the highest non ground truth token. We use these to compute
        AUM and then DEP.

        Note: we are given logprobs, but AUM/DEP requires probs. Rather than
        immediately taking exp(logprobs), we first extract the label logprobs and
        the max non label logprobs to optimize compute - this works because log
        is a monotonic function, so exp(max(logprobs)) = max(probs).

        Parameters:
            sample_logprobs: shape = [seq_len, Vocab size]
            labels: shape = [seq_len, 1]
                Note - at times we need to squeeze the final dimension when
                working with labels

        Returns:
            token_deps: per token DEP - dep.shape = labels.shape
        """
        # TODO: if labels is longer than probs we are in trouble!
        #  this can happen is the user somehow tokenizes their data
        #  differently than the default - e.g. changing max_token_length
        # Re-shape for now to avoid errors
        labels = labels[..., None]
        gold_logprobs = np.take_along_axis(sample_logprobs, labels, axis=-1).squeeze()
        logprobs_copy = sample_logprobs.copy()
        # Logprobs are always < 0 so set to -inf to avoid in max
        logprobs_copy[np.arange(len(labels)), labels.squeeze()] = -float("Inf")
        # Max non-gold logprob
        max_non_gold_logprobs = np.max(logprobs_copy, axis=-1)

        # Convert to probs
        gold_probs = np.exp(gold_logprobs)
        max_non_gold_probs = np.exp(max_non_gold_logprobs)

        # Compute AUM --> DEP
        token_margins = gold_probs - max_non_gold_probs
        token_deps = (1 - token_margins) / 2
        return token_deps

    def process_logprobs(
        self, batch_ids: np.ndarray, batch_logprobs: np.ndarray
    ) -> Tuple[pa.array, pa.array, pa.array, List[float], List[float]]:
        """Handle processing of a batch of sample logprobs

        For each sample in the batch extract / compute the following values:
            - Token level logprobs for the GT label
            - Token level top-k model logprobs: represented as a dictionary
            mapping {predicted_token: logprob}
            - Token level DEP scores
            - Sample level DEP score
            - Sample level model Perplexity

        batch_logprobs has shape - [batch_size, max_token_length, vocab_size], where
        max_token_length is determined by the longest sample in the batch. Because
        other samples in the batch are padded to this max_length, we have to process
        each sample individually to ignore pad token indices.

        Special points of consideration:
            - For each sample, top-k logprobs is a list of dictionaries with length
            equal to the number of tokens in that sample. Each dictionary maps the
            models top-k predicted tokens to their corresponding logprobs.

            - Sample level DEP is computed as the average per token DEP.

            - Perplexity is computed as the exponential of the Cross Entropy loss for
            a sample = -avg(gold_logprob(token_i)). In words, this is the sum of the
            negative logprob of the GT label for each token.
            Reference: https://en.wikipedia.org/wiki/Perplexity

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
            batch_token_deps: The DEP per token prediction
                len(batch_dep) == batch_size
                batch_token_dep[i].shape is [num_tokens_in_label[i]]
            batch_deps: Per sample DEP
                len(batch_deps) == batch_size
            batch_perplexities: Per sample Perplexity
                len(batch_perplexities) == batch_size
        """
        # Compute the top-k logprob indices across the batch.
        top_logprob_indices = get_top_logprob_indices(batch_logprobs)

        batch_token_logprobs = []
        batch_top_logprobs = []
        batch_token_deps = []
        batch_deps = []
        batch_perplexities = []
        # Iterate through the samples in the batch
        for sample_id, sample_logprobs, sample_top_indices in zip(
            batch_ids, batch_logprobs, top_logprob_indices
        ):
            sample_labels = self._retrieve_sample_labels(sample_id)
            (
                sample_logprobs,
                sample_top_indices,
            ) = self._remove_padding(sample_labels, sample_logprobs, sample_top_indices)
            (
                token_logprobs,
                top_logprobs,
            ) = process_sample_logprobs(
                sample_logprobs=sample_logprobs,
                sample_labels=sample_labels,
                sample_top_indices=sample_top_indices,
                tokenizer=self.logger_config.tokenizer,
            )

            batch_token_logprobs.append(token_logprobs)
            batch_top_logprobs.append(top_logprobs)

            # TODO: Likely move to runners - but keep now for backwards compatability
            token_deps = self.compute_token_deps(sample_logprobs, sample_labels)
            batch_token_deps.append(token_deps)
            # Compute sample DEP as avg(token_DEP)
            sample_dep = np.mean(token_deps)
            batch_deps.append(sample_dep)
            # Perplexity = exp(-sum(gold_logprobs)
            perplexity = np.exp(-1 * np.mean(token_logprobs))
            batch_perplexities.append(perplexity)

        return (
            pa.array(batch_token_logprobs),
            pa.array(batch_top_logprobs, type=TOP_LOGPROBS_SCHEMA),
            pa.array(batch_token_deps),
            batch_deps,
            batch_perplexities,
        )

    def _get_data_dict(self) -> Dict:
        """Returns the data dictionary for writing to disk"""
        batch_size = len(self.ids)
        data = {
            C.id.value: self.ids,
            C.token_deps.value: self.token_deps,
            C.dep.value: self.sample_dep,
            C.perplexity.value: self.sample_perplexity,
            C.token_logprobs.value: self.token_logprobs,
            C.top_logprobs.value: self.top_logprobs,
            C.split_.value: [Split[self.split].value] * batch_size,
            C.epoch.value: [self.epoch] * batch_size,
        }
        if self.split == Split.inference:
            data[C.inference_name.value] = [self.inference_name] * batch_size
        return data

    def _write_dict_to_disk(self, path: str, object_name: str, data: Dict) -> None:
        save_arrow_file(path, object_name, data)

    def _retrieve_sample_labels(self, sample_id: int) -> np.ndarray:
        """Retrieve the labels array based on the sample it"""
        labels = np.array(
            self.logger_config.id_to_tokens[self.token_map_key][sample_id]
        )
        return labels

    def _remove_padding(
        self,
        labels: np.ndarray,
        sample_logprobs: np.ndarray,
        sample_top_indices: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Remove padding tokens from logprobs and top_indices

        To remove padding we use the tokenized labels and slice
        tokens depending on the padding side of the tokenizer.

        Parameters:
        -----------

        """
        # Remove padding based on the padding_side of the tokenizer
        num_tokens = len(labels)
        if self.logger_config.tokenizer.padding_side == "left":  # type: ignore
            logprobs = sample_logprobs[-num_tokens:]
            top_indices = sample_top_indices[-num_tokens:]
        else:
            logprobs = sample_logprobs[:num_tokens]
            top_indices = sample_top_indices[:num_tokens]

        return logprobs, top_indices

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
