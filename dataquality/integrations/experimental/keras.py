import inspect
from functools import partial
from typing import Any, Dict, List, Tuple, Union

import tensorflow as tf
from keras.engine import data_adapter
from tensorflow import keras

import dataquality as dq
from dataquality import config
from dataquality.analytics import Analytics
from dataquality.clients.api import ApiClient
from dataquality.exceptions import GalileoException
from dataquality.schemas.split import Split
from dataquality.utils.helpers import wrap_fn
from dataquality.utils.tf import is_tf_2

# If this is TF 1.x
if not is_tf_2():
    tf.compat.v1.enable_eager_execution()

import tensorflow as tf
from keras.callbacks import Callback
from keras.layers import Layer

a = Analytics(ApiClient, config)
a.log_import("integrations/experimental/keras")


def pass_on(*args, **kwargs):
    return None


# https://github.com/tensorflow/tensorflow/issues/33478#issuecomment-843720638
def proxy_call(input: tf.Tensor, obj: tf.keras.layers.Layer, *args, **kwargs):
    if obj._before_call is not None:
        obj._before_call(obj, input)
    output = obj._old_call(input)
    if obj._after_call is not None:
        hook_result = obj._after_call(obj, input, output)
        if hook_result is not None:
            output = hook_result
    return output


def hook_layer(layer, before_call=None, after_call=None):
    layer._before_call = before_call
    layer._after_call = after_call
    layer._old_call = layer.call
    layer.call = partial(proxy_call, obj=layer)


def save_input(store, layer: Any, input: Any, *args, **kwargs):
    if input is not None:
        store["input"] = input


def save_output(
    store, layer: tf.keras.layers.Layer, input: tf.Tensor, output: tf.Tensor
):

    if output is not None:
        store["output"] = output


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


class DataQualityCallback(keras.callbacks.Callback):
    helper_data: Any
    model: Any

    def __init__(self, store, model, *args, **kwargs) -> None:
        self.helper_data = store
        a.log_function("keras/dqcallback")
        self.model = model
        super(DataQualityCallback, self).__init__()

    def set_params(self, params):
        self.helper_data["params"] = params
        self.params = params

    def generate_indices(self, x, kwargs):
        e_model = keras.Sequential([])
        e_model.compile(loss="mse", run_eagerly=True)
        data_handler = data_adapter.get_data_handler(
            x=x,
            model=e_model,
            steps_per_execution=self.model._steps_per_execution,
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
            for step, iterator_data in enumerate(iterator):
                self.helper_data[f"epoch_{epoch}_{step}"] = iterator_data

    def generate_kwargs(self):
        signature = inspect.signature(self.model.fit)
        default_params = {
            k: v.default
            for k, v in signature.parameters.items()
            if v.default is not inspect.Parameter.empty
        }
        kwargs = {**default_params}
        kwargs.update(self.helper_data["fit_kwargs"])
        return kwargs

    def on_train_begin(self, logs=None):
        self.helper_data["step"] = 0
        assert self.model.run_eagerly, GalileoException(
            "Model must be compiled with run_eagerly=True"
        )
        X_values = self.helper_data["fit_kwargs"].get(
            "x", self.helper_data["fit_args"][0]
        )
        x = tf.range(len(X_values))
        self.generate_indices(x, self.generate_kwargs())

    def on_epoch_begin(self, epoch: int, logs: Dict) -> None:
        dq.set_epoch(epoch)
        self.helper_data["step"] = 0

    def _clear_logger_config_helper_data(self) -> None:
        self.helper_data.pop("input", None)
        self.helper_data.pop("output", None)
        self.helper_data.pop("indices_ids", None)

    def on_train_batch_begin(self, batch: Any, logs: Dict = None) -> None:
        self._clear_logger_config_helper_data()
        dq.set_split(Split.train)

    def on_train_batch_end(self, batch: Any, logs: Dict = None) -> None:
        dq.set_split(Split.train)
        logger_config = dq.get_data_logger().logger_config
        ids = self.helper_data.get("indices_ids")
        if ids is None:
            epoch = logger_config.cur_epoch
            step = self.helper_data["step"]
            ids = self.helper_data[f"epoch_{epoch}_{step}"]
            self.helper_data["step"] += 1

        dq.log_model_outputs(
            embs=self.helper_data["input"],
            logits=self.helper_data["output"],
            ids=ids,
        )

    def on_test_begin(self, logs=None):
        dq.set_split(Split.test)
        self.helper_data["step"] = 0

    def on_test_batch_begin(self, batch: Any, logs: Dict = None) -> None:
        # TODO: Somehow we should figure out whether this is in .fit
        #  (so really this should be val) or .evaluate (so this should be test)
        self._clear_logger_config_helper_data()
        dq.set_split(Split.test)

    def on_test_batch_end(self, batch: Any, logs: Dict = None) -> None:
        dq.set_split(Split.test)
        logger_config = dq.get_data_logger().logger_config
        ids = self.helper_data.get("indices_ids")
        return
        if ids is None:
            ids = self.helper_data[f"epoch_{logger_config.cur_epoch}"]

        dq.log_model_outputs(
            embs=self.helper_data["input"],
            logits=self.helper_data["output"][:, : len(logger_config.labels)],
            ids=ids,
        )


def store_model_args_kwargs(store, callback):
    def fit_wrapper(orig_func, *args, **kwargs):
        """Stores the indices of the batch"""
        store["fit_args"] = args
        store["fit_kwargs"] = kwargs
        if kwargs.get("callbacks"):
            kwargs["callbacks"].append(callback)
        else:
            kwargs["callbacks"] = [callback]
        return orig_func(*args, **kwargs)

    return fit_wrapper


def store_model_ids(store):
    def train_step_wrapper(orig_func, *args, **kwargs):
        """Stores the indices of the batch"""
        try:
            store["indices_ids"] = args[0][0].pop("id", None)
        except AttributeError:
            pass

        return orig_func(*args, **kwargs)

    return train_step_wrapper


def select_model_layer(model, layer):
    chosen_layer = layer
    if isinstance(chosen_layer, str) is None:
        for model_layer in model.layers:
            if model_layer.name == layer:
                chosen_layer = model_layer
                break
    else:
        for model_layer in model.layers:
            if model_layer.name == "classifier":
                chosen_layer = model_layer
                break
    return chosen_layer


def watch(model: tf.keras.layers.Layer, layer=None, seed=42):
    tf.random.set_seed(seed)
    logger_data = dq.get_model_logger().logger_config.helper_data
    callback = DataQualityCallback(logger_data, model)
    model.fit = wrap_fn(model.fit, store_model_args_kwargs(logger_data, callback))
    model.train_step = wrap_fn(model.train_step, store_model_ids(logger_data))
    model.test_step = wrap_fn(model.test_step, store_model_ids(logger_data))
    model.predict_step = wrap_fn(model.predict_step, store_model_ids(logger_data))
    model.__call__ = wrap_fn(model.__call__, store_model_ids(logger_data))
    chosen_layer = select_model_layer(model, layer)
    hook_layer(
        chosen_layer,
        before_call=partial(save_input, logger_data),
        after_call=partial(save_output, logger_data),
    )
