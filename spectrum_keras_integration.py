import os

os.environ["GALILEO_CONSOLE_URL"] = "https://console.preprod.rungalileo.io"
os.environ["GALILEO_USERNAME"] = "galileo@rungalileo.io"
os.environ["GALILEO_PASSWORD"] = "I5xV1n7$NBYFmmzH$S8LgkJ"

import dataquality as dq

dq.configure()

# Example taken from: https://keras.io/examples/vision/mnist_convnet/
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import dataquality as dq
import dataquality.integrations.keras

# Model / data parameters
num_classes = 10

# Load the data and split it between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

dq.init("text_classification", "computer_vision", "mnist") # 🌕🔭 Galileo


# Convert image data into text data to work with current console
def convert_image_data_to_text(X):
    return [str(X_sample) for X_sample in  X.reshape(len(X), -1).tolist()]
train_text_inputs = convert_image_data_to_text(x_train)
test_text_inputs = convert_image_data_to_text(x_test)


dq.log_data_samples(ids=list(range(len(x_train))), texts=train_text_inputs, labels=y_train, split="training") # 🌕🔭 Galileo
dq.log_data_samples(ids=list(range(len(x_test))), texts=test_text_inputs, labels=y_test, split="test") # 🌕🔭 Galileo

dq.set_labels_for_run(np.unique(y_train)) # 🌕🔭 Galileo


# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

x_train = dq.integrations.keras.add_ids_to_numpy_arr(x_train, range(len(x_train))) # 🌕🔭 Galileo
x_test = dq.integrations.keras.add_ids_to_numpy_arr(x_test, range(len(x_test))) # 🌕🔭 Galileo

input_shape = x_train[0].shape

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

import tensorflow as tf
import dataquality as dq

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
            if inputs.shape[0].value == None: # TF 1.x
                # In this case the tensor is symbolic and has no real information
                inputs = inputs[..., :-1]
            else:
                inputs, ids = split_into_ids_and_numpy_arr(inputs)
                self.helper_data[self.what_to_log] = ids
                pass
        else:
            self.helper_data[self.what_to_log] = inputs
        return inputs

model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        dq.integrations.keras.DataQualityLoggingLayer("ids"), # 🌕🔭 Galileo
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        dq.integrations.keras.DataQualityLoggingLayer("embs"), # 🌕🔭 Galileo
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
        dq.integrations.keras.DataQualityLoggingLayer("probs"), # 🌕🔭 Galileo
    ]
)

model.summary()


batch_size = 128
epochs = 1

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"], run_eagerly=True) # I wasn't expecting run eagerly to have any affect in TF 1.x

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=[x_test, y_test], callbacks=[dq.integrations.keras.DataQualityCallback()]) # 🌕🔭 Galileo

dq.finish() # 🌕🔭 Galileo
