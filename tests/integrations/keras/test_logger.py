from keras import layers, models
from keras.engine import sequential
import tensorflow as tf

# create an embedding layer
layer1 = layers.Embedding(output_dim=2, input_dim=3)
# create a classifier layer that passes the embedding layer
layer2 = layers.Dense(1, activation="sigmoid")
# create a sequential model
model = models.Sequential([layer1, layer2])
layer1.set_weights([tf.constant([[1, 1], [2, 2], [3, 3]])])
model.run_eagerly = True
input_list = tf.constant([[0, 1, 2]], dtype="int32")
outputs = model.predict(input_list)
# set the weights in classifier layer (layer2) so it will always predict 1
layer2.set_weights([tf.constant([[1], [1]]), tf.constant([0])])
