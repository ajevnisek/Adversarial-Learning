import tensorflow as tf
from tensorflow.keras import layers

from common import IMAGE_SIZE, MNIST_CLASSES


class MNISTClassifierWithTemperature(tf.keras.Model):
    def __init__(self, temperature: float = 1.0):
        activation = 'relu'
        img_rows, img_cols, img_colors = IMAGE_SIZE, IMAGE_SIZE, 1
        super(MNISTClassifierWithTemperature, self).__init__(name='')
        self.conv1 = layers.Conv2D(8, kernel_size=(3, 3),
                                input_shape=(img_rows, img_cols, img_colors),
                                activation=activation)
        self.conv2 = layers.Conv2D(8, (3, 3), activation=activation)
        self.max_pool = layers.MaxPooling2D(pool_size=(2, 2))
        self.dropout1 = layers.Dropout(0.25)
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(128, activation=activation)
        self.dropout2 = layers.Dropout(0.5)
        self.fc2 = layers.Dense(MNIST_CLASSES)
        self.temperature = tf.Variable(temperature, dtype=tf.float32)
        self.activation = layers.Activation('softmax', name='y_pred')

    def reset_temperature(self):
        self.temperature = 1

    def call(self, input_tensor, training=False, mask=None):
        x = self.conv1(input_tensor)
        x = self.conv2(x)
        x = self.max_pool(x)

        x = self.dropout1(x, training=training)
        x = self.flatten(x)
        x = self.fc1(x)

        x = self.dropout2(x, training=training)
        x = self.fc2(x)
        x = x / self.temperature
        x = self.activation(x)

        return x


class BinaryClassifierWithTemperature(MNISTClassifierWithTemperature):
    def __init__(self, temperature: float = 1.0):
        activation = 'relu'
        img_rows, img_cols, img_colors = IMAGE_SIZE, IMAGE_SIZE, 1
        super(BinaryClassifierWithTemperature, self).__init__(temperature)
        self.fc2 = layers.Dense(2)
        self.temperature = tf.Variable(temperature, dtype=tf.float32)
        self.activation = layers.Activation('softmax', name='y_pred')

    def reset_temperature(self):
        self.temperature = 1

    def call(self, input_tensor, training=False, mask=None):
        x = self.conv1(input_tensor)
        x = self.conv2(x)
        x = self.max_pool(x)

        x = self.dropout1(x, training=training)
        x = self.flatten(x)
        x = self.fc1(x)

        x = self.dropout2(x, training=training)
        x = self.fc2(x)
        self.logits = x / self.temperature
        x = self.activation(self.logits)

        return x

