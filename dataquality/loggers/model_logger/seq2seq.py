from typing import Dict, List, Optional, Tuple, Union
from uuid import uuid4

import numpy as np
import pyarrow as pa

from dataquality import config
from dataquality.loggers.logger_config.seq2seq import seq2seq_logger_config
from dataquality.loggers.model_logger.base_model_logger import BaseGalileoModelLogger
from dataquality.schemas.split import Split
from dataquality.utils.arrow import save_arrow_file


class Seq2SeqModelLogger(BaseGalileoModelLogger):
    __logger_name__ = "seq2seq"
    logger_config = seq2seq_logger_config

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
        # assert (
        #     self.labels is not None
        # ), "In Seq2Seq, labels must be provided for the batch"
        self.token_dep = pa.array([])
        self.token_gold_probs = pa.array([])

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

        # Ground truth probs, including the padding for ignored labels
        probs = self.convert_logits_to_probs(self.logits)
        token_dep_padded, gold_probs_padded = self.get_token_dep(probs, self.labels)
        self.token_dep, self.token_gold_probs = self.unpad_dep_probs(
            token_dep_padded, gold_probs_padded, self.labels
        )

    def get_token_dep(
        self, probs: np.ndarray, labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extracts DEP per token prediction

        First, extract the probabilities of the GT token label

        Probs is a numpy array of shape [batch_size, max_token_len, vocab_size] where
        for each sample (text input) in the batch, every token of that sample has a
        probability vector of size vocab_size (which can be 30k+).

        Labels is of shape [batch_size, max_token_length], where for each sample, it
        indicates the index into the vocab that the token should be (the token label).

        We use advanced indexing to extract out only the probabilities for the token
        label for each sample, for each batch.

        Then, we get the second highest probabilities per token via similar indexing.

        Finally, compute dep and return.

        Returns: (token_dep, gold_probs)
        """
        batch_size, max_sequence_length, vocab_size = probs.shape
        clean_labels = labels.copy()
        # The labels are set to -100 for ignored tokens. Since the shape is of
        # `max_token_length`, many tokens in a particular sample may be ignored if they
        # don't exist. Similarly, in the case of a decoder-only model, the inputs will
        # be a part of the sample, so the labels are set to -100 so they are ignored
        clean_labels[clean_labels == -100] = 0

        # Create an array of indices for advanced indexing
        batch_indices = np.arange(batch_size)[:, np.newaxis]
        sequence_indices = np.arange(max_sequence_length)[np.newaxis, :]

        # Use advanced indexing to extract the logits for the label tokens
        gold_probs = probs[batch_indices, sequence_indices, clean_labels]

        # Now we set the location of the gold_probs to 0 so we can easily get the
        # second highest, _non_gold_ probs
        probs_no_gold = probs.copy()
        probs_no_gold[batch_indices, sequence_indices, labels] = 0
        # The probability of the second highest for each token in the sample
        second_probs = probs_no_gold.max(axis=-1)
        token_dep = (1 - (gold_probs - second_probs)) / 2
        return token_dep, gold_probs

    def unpad_dep_probs(
        self, token_dep: np.ndarray, token_gold_probs: np.ndarray, labels: np.ndarray
    ) -> Tuple[pa.array, pa.array]:
        """Unpads the incoming numpy array by looking for padded/ignored indices

        Ignored/padded indices are indicated by a -100 in the labels array.

        token_dep, token_gold_probs, and labels are of shape
        [batch_size, max_token_length], but for each sample in the batch, the tokens
        for that sample that are ignored are -100 in the labels matrix.
        So we use that to get only the ones we care about.

        We return a pyarrow array because each batch will have a different shape, which
        can't be represented in numpy
        """
        # batch_num, non_pad_idx = np.where(labels!=100)

        # token_dep[batch_num, non_pad_idx]

        batch_deps = []
        batch_gold_probs = []
        for batch_token_dep, batch_token_probs, batch_labels in zip(
            token_dep, token_gold_probs, labels
        ):
            batch_deps.append(batch_token_dep[batch_labels != -100])
            batch_gold_probs.append(batch_token_probs[batch_labels != -100])

        dep = pa.array(batch_deps)
        gold_probs = pa.array(batch_gold_probs)
        return dep, gold_probs

    def _get_data_dict(self) -> Dict:
        """Returns the data dictionary for writing to disk"""
        # TODO: Do we need to include the labels?
        batch_size = len(self.ids)
        data = {
            "id": self.ids,
            "token_dep": self.token_dep,
            "token_gold_probs": self.token_gold_probs,
            "labels": pa.array(list(self.labels)),
            "split": [Split[self.split].value] * batch_size,
            "epoch": [self.epoch] * batch_size,
        }
        if self.split == Split.inference:
            data["inference_name"] = [self.inference_name] * batch_size
        return data

    def write_model_output(self, data: Dict) -> None:
        """Creates an arrow file from the current batch data"""
        """Creates an hdf5 file from the data dict"""
        location = (
            f"{self.LOG_FILE_DIR}/{config.current_project_id}"
            f"/{config.current_run_id}"
        )
        split = data["split"][0]

        if split == Split.inference:
            inference_name = data["inference_name"][0]
            path = f"{location}/{split}/{inference_name}"
        else:
            epoch = data["epoch"][0]
            path = f"{location}/{split}/{epoch}"

        object_name = f"{str(uuid4()).replace('-', '')[:12]}.hdf5"
        save_arrow_file(path, object_name, data)
