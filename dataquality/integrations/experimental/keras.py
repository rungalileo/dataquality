from typing import Any, Dict, List, Tuple, Union
from dataquality import config

import tensorflow as tf
from tensorflow import keras
from keras.engine import data_adapter
import inspect
import dataquality as dq
from dataquality.analytics import Analytics
from dataquality.clients.api import ApiClient
from functools import partial

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

    def on_train_begin(self, logs=None):
        e_model = keras.Sequential([])
        e_model.compile(loss="mse", run_eagerly=True)
        assert self.model.run_eagerly, GalileoException(
            "Model must be compiled with run_eagerly=True"
        )
        X_values = self.helper_data["fit_kwargs"].get(
            "x", self.helper_data["fit_args"][0]
        )
        x = tf.range(len(X_values))
        signature = inspect.signature(self.model.fit)
        default_params = {
            k: v.default
            for k, v in signature.parameters.items()
            if v.default is not inspect.Parameter.empty
        }
        kwargs = {**default_params}
        kwargs.update(self.helper_data["fit_kwargs"])
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
            iterator_data = iterator.get_next()
            self.helper_data[f"epoch_{epoch}"] = iterator_data

    def on_epoch_begin(self, epoch: int, logs: Dict) -> None:
        dq.set_epoch(epoch)

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
            ids = self.helper_data[f"epoch_{logger_config.cur_epoch}"]

        dq.log_model_outputs(
            embs=self.helper_data["input"],
            logits=self.helper_data["output"][:, : len(logger_config.labels)],
            ids=ids,
        )

    def on_test_batch_begin(self, batch: Any, logs: Dict = None) -> None:
        # TODO: Somehow we should figure out whether this is in .fit
        #  (so really this should be val) or .evaluate (so this should be test)
        self._clear_logger_config_helper_data()
        dq.set_split(Split.test)

    def on_test_batch_end(self, batch: Any, logs: Dict = None) -> None:
        dq.set_split(Split.test)
        logger_config = dq.get_data_logger().logger_config
        ids = self.helper_data.get("indices_ids")
        if ids is None:
            ids = self.helper_data[f"epoch_{logger_config.cur_epoch}"]

        dq.log_model_outputs(
            embs=self.helper_data["input"],
            logits=self.helper_data["output"][:, : len(logger_config.labels)],
            ids=ids,
        )


def proxy_call(input, obj, *args, **kwargs):
    if obj._before_call is not None:
        obj._before_call(obj, input)
    output = obj._old_call(input)
    if obj._after_call is not None:
        hook_result = obj._after_call(obj, input, output)
        if hook_result is not None:
            output = hook_result
    return output


def store_model_args_kwargs(store, callback):
    def fit_wrapper(orig_func, *args, **kwargs):
        """Stores the indices of the batch"""
        store["fit_args"] = args
        store["fit_kwargs"] = kwargs
        if kwargs.get("callbacks"):
            kwargs["callbacks"].append(callback)
        else:
            kwargs["callbacks"] = [callback]
        indices = orig_func(*args, **kwargs)
        if indices:
            store["ids"] = indices
        return indices

    return fit_wrapper


def store_model_ids(store):
    def train_step_wrapper(orig_func, *args, **kwargs):
        """Stores the indices of the batch"""
        store["indices_ids"] = args[0][0].pop("id", None)

        return orig_func(*args, **kwargs)

    return train_step_wrapper


def watch(model, seed=42):
    tf.random.set_seed(seed)
    logger_data = dq.get_model_logger().logger_config.helper_data
    callback = DataQualityCallback(logger_data, model)
    model.fit = wrap_fn(model.fit, store_model_args_kwargs(logger_data, callback))
    model.train_step = wrap_fn(model.train_step, store_model_ids(logger_data))
    model.test_step = wrap_fn(model.test_step, store_model_ids(logger_data))
    model.predict_step = wrap_fn(model.predict_step, store_model_ids(logger_data))
    model.__call__ = wrap_fn(model.__call__, store_model_ids(logger_data))
    hook_layer_call(
        model.layers,
        before_call=partial(save_input, logger_data),
        after_call=partial(save_output, logger_data),
    )
