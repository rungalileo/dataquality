from typing import Any, Dict, List, Tuple, Union

import numpy as np
import tensorflow as tf
from tensorflow import keras

import dataquality as dq
from dataquality.exceptions import GalileoException
from dataquality.schemas.split import Split
from dataquality.utils.tf import is_tf_2

# If this is TF 1.x
if not is_tf_2():
    tf.compat.v1.enable_eager_execution()


def _indices_for_ids(arr: np.ndarray) -> Tuple:
    return tuple([list(range(arr.shape[0]))] + [[-1]] * (len(arr.shape) - 1))


def add_ids_to_numpy_arr(
    orig_arr: np.ndarray, ids: Union[List[int], np.ndarray]
) -> np.ndarray:
    arr = np.concatenate([orig_arr, np.zeros(orig_arr.shape[:-1] + (1,))], axis=-1)
    arr[_indices_for_ids(arr)] = ids
    return arr


def split_into_ids_and_numpy_arr(arr: tf.Tensor) -> Tuple[tf.Tensor, np.ndarray]:
    orig_arr = arr[..., :-1]
    # Tflow doesn't seem to quite support my advanced indexing
    if tf.is_tensor(arr):
        arr = arr.numpy()
    ids = arr[_indices_for_ids(arr)]
    return orig_arr, ids.astype(int)


class DataQualityLoggingLayer(tf.keras.layers.Layer):
    def __init__(self, what_to_log: str):
        super(DataQualityLoggingLayer, self).__init__()
        if what_to_log not in ["ids", "probs", "embs"]:
            raise GalileoException("What to log must be one of ids, probs or embs")
        self.what_to_log = what_to_log
        self.helper_data = dq.get_model_logger().logger_config.helper_data

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        if self.what_to_log == "ids":
            is_input_symbolic = False
            if is_tf_2():
                is_input_symbolic = inputs.shape[0] is None
            else:
                is_input_symbolic = inputs.shape[0].value is None

            if is_input_symbolic:
                inputs = inputs[..., :-1]
            else:
                inputs, ids = split_into_ids_and_numpy_arr(inputs)
                self.helper_data[self.what_to_log] = ids
        else:
            self.helper_data[self.what_to_log] = inputs
        return inputs


class DataQualityCallback(keras.callbacks.Callback):
    def __init__(self) -> None:
        super(DataQualityCallback, self).__init__()
        self.helper_data = dq.get_model_logger().logger_config.helper_data
        # In the future we could maybe insert the layers into sequential or something

    def on_train_begin(self, logs: Dict) -> None:
        dq.set_split(Split.train)

    def on_test_begin(self, logs: Dict) -> None:
        # TODO: Somehow we should figure out whether this is in .fit
        #  (so really this should be val) or .evaluate (so this should be test)
        dq.set_split(Split.test)

    def on_epoch_begin(self, epoch: int, logs: Dict) -> None:
        dq.set_epoch(epoch)
        print(f"Starting with epoch {epoch}")

    def _clear_logger_config_helper_data(self) -> None:
        self.helper_data["embs"] = None
        self.helper_data["probs"] = None
        self.helper_data["logits"] = None

    def on_train_batch_begin(self, batch: Any, logs: Dict = None) -> None:
        self._clear_logger_config_helper_data()

    def on_train_batch_end(self, batch: Any, logs: Dict = None) -> None:
        dq.get_model_logger()(**self.helper_data).log()

    def on_test_batch_begin(self, batch: Any, logs: Dict = None) -> None:
        self._clear_logger_config_helper_data()

    def on_test_batch_end(self, batch: Any, logs: Dict = None) -> None:
        dq.get_model_logger()(**self.helper_data).log()
