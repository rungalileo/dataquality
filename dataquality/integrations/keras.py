import keras
import dataquality as dq
import numpy as np
import tensorflow as tf
from dataquality.utils.tf import is_tf_2

# If this is TF 1.x
if not is_tf_2():
    tf.compat.v1.enable_eager_execution()

def _indices_for_ids(arr):
    return tuple([list(range(arr.shape[0]))] + [[-1]] * (len(arr.shape) - 1))


def add_ids_to_numpy_arr(orig_arr, ids):
    arr = np.concatenate([orig_arr, np.zeros(orig_arr.shape[:-1] + (1,))], axis=-1)
    arr[_indices_for_ids(arr)] = ids
    return arr


def split_into_ids_and_numpy_arr(arr):
    orig_arr = arr[..., :-1]
    # Tflow doesn't seem to quite support my advanced indexing
    if tf.is_tensor(arr):
        arr = arr.numpy()
    ids = arr[_indices_for_ids(arr)]
    return orig_arr, ids.astype(int)


class DataQualityLoggingLayer(tf.keras.layers.Layer):
    def __init__(self, what_to_log: str):
        super(DataQualityLoggingLayer, self).__init__()
        self.what_to_log = what_to_log
        self.helper_data = dq.get_model_logger().logger_config.helper_data

    def call(self, inputs):
        if self.what_to_log == "ids":
            is_input_symbolic = False
            if is_tf_2():
                is_input_symbolic = inputs.shape[0] == None
            else:
                is_input_symbolic = inputs.shape[0].value == None

            if is_input_symbolic:
                inputs = inputs[..., :-1]
            else:
                inputs, ids = split_into_ids_and_numpy_arr(inputs)
                self.helper_data[self.what_to_log] = ids
        else:
            self.helper_data[self.what_to_log] = inputs
        return inputs


# For more info see: https://keras.io/guides/writing_your_own_callbacks/#usage-of-selfmodel-attribute
import dataquality as dq


class DataQualityCallback(keras.callbacks.Callback):
    def __init__(self):
        super(DataQualityCallback, self).__init__()
        self.helper_data = dq.get_model_logger().logger_config.helper_data
        # In the future we could maybe insert the layers into sequential or something

    def on_train_begin(self, logs):
        dq.set_split("train")

    def on_test_begin(self, logs):
        # TODO: Somehow we should figure out whether this is in .fit (so really this should be val) or .evaluate (so this should be test)
        dq.set_split("test")

    def on_epoch_begin(self, epoch, logs):
        dq.set_epoch(epoch)
        print(f"Starting with epoch {epoch}")

    def _clear_logger_config_helper_data(self):
        self.helper_data["embs"] = None
        self.helper_data["probs"] = None
        self.helper_data["logits"] = None

    def on_train_batch_begin(self, batch, logs=None):
        self._clear_logger_config_helper_data()

    def on_train_batch_end(self, batch, logs=None):
        dq.get_model_logger()(**self.helper_data).log()

    def on_test_batch_begin(self, batch, logs=None):
        self._clear_logger_config_helper_data()

    def on_test_batch_end(self, batch, logs=None):
        dq.get_model_logger()(**self.helper_data).log()

