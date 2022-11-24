import inspect
from functools import partial
from typing import Any, Callable, Dict, Optional, Sized, Tuple, Union

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

a = Analytics(ApiClient, config)
a.log_import("integrations/experimental/keras")


def pass_on(*args: Tuple, **kwargs: Dict[str, Any]) -> None:
    return None


# https://github.com/tensorflow/tensorflow/issues/33478#issuecomment-843720638
def proxy_call(
    input: tf.Tensor, obj: tf.keras.layers.Layer, *args: Tuple, **kwargs: Dict[str, Any]
) -> Any:
    if obj._before_call is not None:
        obj._before_call(obj, input)
    output = obj._old_call(input)
    if obj._after_call is not None:
        hook_result = obj._after_call(obj, input, output)
        if hook_result is not None:
            output = hook_result
    return output


def hook_layer(
    layer: tf.keras.layers.Layer,
    before_call: Optional[Callable] = None,
    after_call: Optional[Callable] = None,
) -> None:
    layer._before_call = before_call
    layer._after_call = after_call
    layer._old_call = layer.call
    layer.call = partial(proxy_call, obj=layer)


def save_input(
    store: Dict[str, Any],
    layer: tf.keras.layers.Layer,
    input: tf.Tensor,
    *args: Tuple,
    **kwargs: Dict[str, Any],
) -> None:
    if input is not None:
        store["input"] = input


def save_output(
    store: Dict[str, Any],
    layer: tf.keras.layers.Layer,
    input: tf.Tensor,
    output: tf.Tensor,
) -> None:

    if output is not None:
        store["output"] = output


class DataQualityCallback(keras.callbacks.Callback):
    helper_data: Any
    model: Any

    def __init__(
        self,
        store: Dict[str, Any],
        model: tf.keras.layers.Layer,
        *args: Tuple,
        **kwargs: Dict[str, Any],
    ) -> None:
        self.helper_data = store
        a.log_function("keras/dqcallback")
        self.model = model
        super(DataQualityCallback, self).__init__()
        if not tf.executing_eagerly():
            raise GalileoException("Needs to be executing eagerly")

    def generate_test_indices(self, kwargs: Dict[str, Any]) -> None:
        e_model = keras.Sequential([])
        e_model.compile(loss="mse", run_eagerly=True)
        data_handler = data_adapter.DataHandler(
            x=kwargs.get("val_x"),
            y=None,
            sample_weight=kwargs.get("val_sample_weight"),
            batch_size=kwargs.get("validation_batch_size") or kwargs.get("batch_size"),
            steps_per_epoch=kwargs.get("validation_steps"),
            initial_epoch=0,
            epochs=1,
            max_queue_size=kwargs.get("max_queue_size"),
            workers=1,
            use_multiprocessing=False,
            model=e_model,
            steps_per_execution=self.model._steps_per_execution,
        )
        for epoch, iterator in data_handler.enumerate_epochs():
            for step, iterator_data in enumerate(iterator):
                self.helper_data[f"val_{step}"] = iterator_data

    def generate_train_indices(self, kwargs: Dict[str, Any]) -> None:
        e_model = keras.Sequential([])
        e_model.compile(loss="mse", run_eagerly=True)
        filtered_keys = set(kwargs.keys())
        filtered_keys.discard("model")
        filtered_keys.discard("steps_per_execution")
        allowed_keys = set(inspect.signature(data_adapter.DataHandler).parameters)

        keys = filtered_keys.intersection(allowed_keys)
        dh_kwargs = {key: kwargs.get(key) for key in keys}
        data_handler = data_adapter.get_data_handler(
            model=e_model,
            steps_per_execution=self.model._steps_per_execution,
            **dh_kwargs,
        )

        for epoch, iterator in data_handler.enumerate_epochs():
            for step in data_handler.steps():
                self.helper_data[f"epoch_{epoch}_{step}"] = next(iterator)

    def generate_kwargs(
        self, func: Callable, fit_kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        signature = inspect.signature(func)
        default_params = {
            k: v.default
            for k, v in signature.parameters.items()
            if v.default is not inspect.Parameter.empty
        }
        kwargs = {**default_params}
        kwargs.update(fit_kwargs)
        return kwargs

    def on_train_begin(self, logs: Any = None) -> None:
        assert self.model.run_eagerly, GalileoException(
            "Model must be compiled with run_eagerly=True"
        )
        fit_kwargs = self.helper_data["model_fit"].get("kwargs")
        fit_args = self.helper_data["model_fit"].get("args")
        self.fit_kwargs = self.generate_kwargs(self.model.fit, fit_kwargs)
        x = self.fit_kwargs.get("x")
        if x is None and len(fit_args):
            # TODO: enumerate args and replace kwargs
            x = fit_args[0]
        if isinstance(x, Sized):
            x_len = len(x)
        else:
            raise GalileoException("Can not determine length of x")
        new_kwargs = self.generate_split(self.fit_kwargs, x_len)
        self.fit_kwargs.update(new_kwargs)
        self.generate_train_indices(self.fit_kwargs)

    def generate_split(self, kwargs: Dict[str, Any], x_len: int) -> Dict:
        validation_split = kwargs.get("validation_split")
        validation_data = kwargs.get("validation_data")
        sample_weight = kwargs.get("sample_weight")
        x = tf.range(x_len)
        y = x

        if validation_split and validation_data is None:
            # Create the validation data using the training data. Only supported
            # for `Tensor` and `NumPy` input.
            (
                x,
                y,
                sample_weight,
            ), validation_data = data_adapter.train_validation_split(
                (x, y, sample_weight), validation_split=validation_split
            )

        if validation_data:
            (
                val_x,
                val_y,
                val_sample_weight,
            ) = data_adapter.unpack_x_y_sample_weight(validation_data)
            val_x = tf.range(len(val_x))
        else:
            val_x = None
            val_sample_weight = None
        return {
            "x": x,
            "y": None,
            "sample_weight": sample_weight,
            "val_x": val_x,
            "val_y": None,
            "val_sample_weight": val_sample_weight,
        }

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
            epoch = logger_config.cur_epoch
            step = batch
            ids = self.helper_data[f"epoch_{epoch}_{step}"]
        dq.log_model_outputs(
            embs=self.helper_data["input"],
            logits=self.helper_data["output"],
            ids=ids,
        )

    def on_test_begin(self, logs: Any = None) -> None:
        dq.set_split(Split.test)
        self.generate_test_indices(self.fit_kwargs)

    def on_test_batch_begin(self, batch: Any, logs: Dict = None) -> None:
        # TODO: Somehow we should figure out whether this is in .fit
        #  (so really this should be val) or .evaluate (so this should be test)
        self._clear_logger_config_helper_data()
        dq.set_split(Split.validation)

    def on_test_batch_end(self, batch: Any, logs: Dict = None) -> None:
        dq.set_split(Split.validation)
        # logger_config = dq.get_data_logger().logger_config
        ids = self.helper_data.get("indices_ids")

        if ids is None:
            step = batch
            ids = self.helper_data[f"val_{step}"]
        dq.log_model_outputs(
            embs=self.helper_data["input"],
            logits=self.helper_data["output"],
            ids=ids,
        )

    def on_predict_begin(self, batch: int) -> None:
        dq.set_split(Split.test)
        predict_kwargs = self.helper_data["model_predict"].get("kwargs")
        predict_args = self.helper_data["model_predict"].get("args")
        self.predict_kwargs = self.generate_kwargs(self.model.predict, predict_kwargs)
        x = self.predict_kwargs.get("x")
        if x is None and len(predict_args):
            # TODO: enumerate args and replace kwargs
            x = predict_args[0]
        if isinstance(x, Sized):
            x_len = len(x)
        else:
            raise GalileoException("Can not determine length of x")
        e_model = keras.Sequential([])
        e_model.compile(loss="mse", run_eagerly=True)
        data_handler = data_adapter.DataHandler(
            x=tf.range(x_len),
            batch_size=self.predict_kwargs.get("batch_size"),
            steps_per_epoch=self.predict_kwargs.get("steps"),
            initial_epoch=0,
            epochs=1,
            max_queue_size=self.predict_kwargs.get("max_queue_size"),
            workers=1,
            use_multiprocessing=False,
            model=e_model,
            steps_per_execution=self.model._steps_per_execution,
        )
        for epoch, iterator in data_handler.enumerate_epochs():
            for step, iterator_data in enumerate(iterator):
                self.helper_data[f"test_{step}"] = iterator_data

    def on_predict_batch_begin(self, batch: int, logs: Any = None) -> None:
        dq.set_epoch(0)

    def on_predict_batch_end(self, batch: int, logs: Any = None) -> None:
        dq.set_split(Split.test)
        # logger_config = dq.get_data_logger().logger_config
        ids = self.helper_data.get("indices_ids")

        if ids is None:
            step = batch
            ids = self.helper_data[f"test_{step}"]

        dq.log_model_outputs(
            embs=self.helper_data["input"],
            logits=self.helper_data["output"],
            ids=ids,
        )


def store_args_kwargs(store: Dict[str, Any], callback: Callable) -> Callable:
    def fit_wrapper(orig_func: Callable, *args: Any, **kwargs: Any) -> None:
        """Stores the indices of the batch"""
        store["args"] = args
        store["kwargs"] = kwargs
        if kwargs.get("callbacks"):
            kwargs["callbacks"].append(callback)
        else:
            kwargs["callbacks"] = [callback]
        return orig_func(*args, **kwargs)

    return fit_wrapper


def store_model_ids(store: Dict[str, Any]) -> Callable:
    def train_step_wrapper(orig_func: Callable, *args: Any, **kwargs: Any) -> None:
        """Stores the indices of the batch"""
        try:
            store["indices_ids"] = args[0][0].pop("id", None)
        except AttributeError:
            pass

        return orig_func(*args, **kwargs)

    return train_step_wrapper


def select_model_layer(
    model: tf.keras.layers.Layer, layer: Optional[Union[tf.keras.layers.Layer, str]]
) -> tf.keras.layers.Layer:
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
    assert chosen_layer is not None, GalileoException("Layer could not be found")
    assert not isinstance(chosen_layer, str), GalileoException(
        "Layer could not be found"
    )
    return chosen_layer


def watch(model: tf.keras.layers.Layer, layer: Any = None, seed: int = 42) -> None:
    tf.random.set_seed(seed)
    logger_data = dq.get_model_logger().logger_config.helper_data
    callback = DataQualityCallback(logger_data, model)
    logger_data["model_fit"] = {}

    logger_data["model_patches"] = {
        "predict": model.predict,
        "test_step": model.test_step,
        "predict_step": model.predict_step,
        "__call__": model.__call__,
        "fit": model.fit,
    }

    model.fit = wrap_fn(
        model.fit, store_args_kwargs(logger_data["model_fit"], callback)
    )
    logger_data["model_predict"] = {}

    model.predict = wrap_fn(
        model.predict, store_args_kwargs(logger_data["model_predict"], callback)
    )

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


def unwatch(model: tf.keras.layers.Layer) -> None:
    logger_data = dq.get_model_logger().logger_config.helper_data
    for k, v in logger_data.get("model_patches", {}).items():
        print("unpatching", k)
        setattr(model, k, v)

    for layer in model.layers:
        if hasattr(layer, "_old_call"):
            layer.call = layer._old_call
            del layer._old_call
            del layer._before_call
            del layer._after_call
