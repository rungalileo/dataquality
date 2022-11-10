import warnings
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import tensorflow as tf
from tensorflow import keras

import dataquality as dq
from dataquality.analytics import Analytics
from dataquality.clients.api import ApiClient

# from dataquality.analytics import Analytics
from dataquality.exceptions import GalileoException
from dataquality.schemas.split import Split
from dataquality.utils.tf import is_tf_2

a = Analytics(ApiClient, dq.config)
a.log_import("keras")

# If this is TF 1.x
if not is_tf_2():
    tf.compat.v1.enable_eager_execution()


def _indices_for_ids(arr: np.ndarray) -> Tuple:
    return tuple([list(range(arr.shape[0]))] + [[-1]] * (len(arr.shape) - 1))


def add_ids_to_numpy_arr(
    orig_arr: np.ndarray, ids: Union[List[int], np.ndarray]
) -> np.ndarray:
    """Deprecated, use add_sample_ids"""
    warnings.warn("Deprecated, use add_sample_ids", DeprecationWarning, stacklevel=2)
    return add_sample_ids(orig_arr, ids)


def add_sample_ids(
    orig_arr: np.ndarray, ids: Union[List[int], np.ndarray]
) -> np.ndarray:
    """Add sample IDs to the training/test data before training begins

    This is necessary to call before training a Keras model with the
    Galileo DataQualityCallback

    :param orig_arr: The numpy array to be passed into model.train
    :param ids: The ids for each sample to append. These are the same IDs that are
    logged for the input data. They must match 1-1
    """
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

            # Sometimes a "symbolic" input is fed in for testing. This is not a real
            # sample. We don't want to save that sample as real IDs, just pass it
            # through and extract out the ID layer
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
        a.log_function("keras/dqcallback")
        super(DataQualityCallback, self).__init__()
        self.helper_data = dq.get_model_logger().logger_config.helper_data
        # In the future we could maybe insert the layers into sequential or something

    def on_epoch_begin(self, epoch: int, logs: Dict) -> None:
        dq.set_epoch(epoch)

    def _clear_logger_config_helper_data(self) -> None:
        self.helper_data.clear()

    def on_train_batch_begin(self, batch: Any, logs: Dict = None) -> None:
        self._clear_logger_config_helper_data()
        dq.set_split(Split.train)

    def on_train_batch_end(self, batch: Any, logs: Dict = None) -> None:
        dq.log_model_outputs(**self.helper_data)

    def on_test_batch_begin(self, batch: Any, logs: Dict = None) -> None:
        # TODO: Somehow we should figure out whether this is in .fit
        #  (so really this should be val) or .evaluate (so this should be test)
        self._clear_logger_config_helper_data()
        dq.set_split(Split.test)

    def on_test_batch_end(self, batch: Any, logs: Dict = None) -> None:
        dq.log_model_outputs(**self.helper_data)


# try:
#     Analytics().log("import", "dataquality.keras")
# except Exception:
#     pass
