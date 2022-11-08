import warnings
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import tensorflow as tf
from tensorflow import keras
import inspect
import dataquality as dq
from dataquality.analytics import Analytics
from dataquality.clients.api import ApiClient
from functools import partial


# from dataquality.analytics import Analytics
from dataquality.exceptions import GalileoException
from dataquality.schemas.split import Split
from dataquality.utils.tf import is_tf_2

# If this is TF 1.x
if not is_tf_2():
    tf.compat.v1.enable_eager_execution()

import tensorflow as tf
from keras.callbacks import Callback
from keras.layers import Layer

# Inspiration:
# https://pytorch.org/vision/stable/feature_extraction.html
# https://github.com/archinetai/surgeon-pytorch/blob/main/surgeon_pytorch/inspect.py


# Clues:
# https://github.com/tensorflow/tensorflow/issues/33129
# earlyPredictor = tf.keras.Model(dcnn.inputs,dcnn.get_layer(theNameYouWant).output).
# model_output = mobilenet_model.get_layer("conv_pw_13_relu").output
# m = Model(inputs=mobilenet_model.input, outputs=model_output)
# https://stackoverflow.com/questions/49492255/how-to-replace-or-insert-intermediate-layer-in-keras-model
# https://github.com/tensorflow/tensorflow/issues/33478
# https://stackoverflow.com/questions/46526869/keras-tensors-get-values-with-indices-coming-from-another-tensor
# https://stackoverflow.com/questions/72503769/is-there-a-tensorflow-function-for-finding-the-indices-next-to-a-condition
class ModelHooker(Callback):
    """
    ```python
       callbacks =  tf.keras.callbacks.CallbackList([...])
       callbacks.append(...)
       callbacks.on_train_begin(...)
       for epoch in range(EPOCHS):
         callbacks.on_epoch_begin(epoch)
         for i, data in dataset.enumerate():
           callbacks.on_train_batch_begin(i)
           batch_logs = model.train_step(data)
           callbacks.on_train_batch_end(i, batch_logs)
         epoch_logs = ...
         callbacks.on_epoch_end(epoch, epoch_logs)
       final_logs=...
       callbacks.on_train_end(final_logs)
    ```
    """

    if not tf.executing_eagerly():
        if tf.inside_function():
            raise ValueError(
                "This Callback's method contains Python state and "
                "should be called outside of `tf.function`s."
            )
        else:  # Legacy graph mode:
            raise ValueError(
                "BackupAndRestore only supports eager mode. In graph "
                "mode, consider using ModelCheckpoint to manually save "
                "and restore weights with `model.load_weights()` and by "
                "providing `initial_epoch` in `model.fit()` for fault "
                "tolerance."
            )

    def __init__(self, model, layer, embedding_dims, logits_dims, **kwargs):
        self.model = model
        self.index_model = tf.keras.Sequential()


class DataQualityLoggingLayer(tf.keras.layers.Layer):
    store: Any

    def __init__(self, store: Any):
        super(DataQualityLoggingLayer, self).__init__()
        self.store = store
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
    store: Any
    model: Any

    def __init__(self, store, *args, **kwargs) -> None:
        self.store = store
        a.log_function("keras/dqcallback")
        super(DataQualityCallback, self).__init__()
        self.helper_data = dq.get_model_logger().logger_config.helper_data
        # In the future we could maybe insert the layers into sequential or something

    def set_params(self, params):
        print(params)
        self.store["params"] = params
        self.params = params

    def on_train_begin(self, logs=None):
        e_model = keras.Sequential([DataQualityLoggingLayer(logger_data["quiet"])])
        e_model.compile(loss="mse", run_eagerly=True)
        x = tf.range(len(self.store["kwargs"]["x"]))
        signature = inspect.signature(self.model.fit)
        default_params = {
            k: v.default
            for k, v in signature.parameters.items()
            if v.default is not inspect.Parameter.empty
        }
        kwargs = {**default_params}
        kwargs.update(logger_data["kwargs"])
        data_handler = data_adapter.get_data_handler(
            x=x,
            model=e_model,
            steps_per_execution=vg16._steps_per_execution,
            **{
                key: value
                for key, value in kwargs.items()
                if key
                not in [
                    "validation_freq",
                    "validation_batch_size",
                    "validation_steps",
                    "validation_data",
                    "validation_split",
                    "callbacks",
                    "x",
                    "verbose",
                ]
            },
        )
        for epoch, iterator in data_handler.enumerate_epochs():
            self.store[f"epoch_{epoch}"] = iterator.get_next()

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


def proxy_call(input, obj):
    print("proxy call")
    if obj._before_call is not None:
        obj._before_call(obj, input)
    output = obj._old_call(input)
    if obj._after_call is not None:
        hook_result = obj._after_call(obj, input, output)
        if hook_result is not None:
            output = hook_result
    return output


def pass_on(*args, **kwargs):
    return None


def hook_layer_call(layers, before_call=None, after_call=None):
    len_layers = len(layers)
    for i, layer in enumerate(layers):
        if i == 1:
            layer._before_call = before_call
            layer._after_call = pass_on
            layer._old_call = layer.call
            layer.call = partial(proxy_call, obj=layer)
        if i == len_layers - 1:
            layer._before_call = pass_on
            layer._after_call = after_call
            layer._old_call = layer.call
            layer.call = partial(proxy_call, obj=layer)


def print_input(store, layer: tf.keras.layers.Layer, input: tf.Tensor):
    print(input.shape)
    store["input"]
    if store["input"] is not None:
        print("saving input")
        store["input"].append(input)


def print_input_output(
    store, layer: tf.keras.layers.Layer, input: tf.Tensor, output: tf.Tensor
):
    print(input.shape, output.shape)
    store["output"]
    if store["output"] is not None:
        print("saving outpu")
        store["output"].append(output)


# suppose you have a model(such as a tf.keras.Sequential instance)
hook_layer_call(
    vg16.layers,
    before_call=partial(print_input, logger_data),
    after_call=partial(print_input_output, logger_data),
)


def store_batch_indices(store):
    def process_batch_indices(next_index_func, *args, **kwargs):
        """Stores the indices of the batch"""

        store["args"] = args
        store["kwargs"] = kwargs
        indices = next_index_func(*args, **kwargs)
        if indices:
            store["ids"] = indices
        return indices

    return process_batch_indices


# %%


my_callbacks = [DataQualityCallback(logger_data)]
vg16 = tf.keras.applications.VGG16()
vg16.compile(optimizer="adam", loss="categorical_crossentropy", run_eagerly=True)
vg16.fit = wrap_fn(vg16.fit, store_batch_indices(logger_data))

# multiply X_ones by X_range to get different values

# https://stackoverflow.com/questions/10093293/is-there-a-python-equivalent-of-rangen-for-multidimensional-ranges
X_ones = tf.ones((64, 224, 224, 3), dtype=tf.float32)
X_range = tf.range(len(X_ones), dtype=tf.float32)
X_range = X_ones * tf.expand_dims(
    tf.expand_dims(tf.expand_dims(X_range, axis=-1), axis=-1), axis=-1
)


# save append in store for key beer

tf.random.set_seed(42)
vg16.fit(
    x=X_range,
    y=tf.ones((len(X_range), 1000)),
    epochs=2,
    batch_size=32,
    callbacks=my_callbacks,
)
# %%
logger_data["input"][1][:, 0, 0, 0]
# %%
logger_data["epoch_0"][0]
logger_data["epoch_1"][0]
