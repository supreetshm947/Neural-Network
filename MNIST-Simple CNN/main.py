import tensorflow as tf
from tensorflow import keras

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

import matplotlib.pyplot as plt

plt.imshow(x_train[0], cmap=plt.cm.binary)

model = tf.keras.models.Sequential()
print(x_train.shape)
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')

model.fit(x_train, y_train, epochs=10)

model.summary()
keras.utils.plot_model(model, "./arch.png", show_shapes=True)

val_loss, val_accuracy = model.evaluate(x_test, y_test)

model.save("mnist_num_reader.model")