import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from gan_trainer import GANTrainer
from common import IMAGE_SIZE, MNIST_CLASSES


class MNISTClassifier(tf.keras.Model):
    def __init__(self):
        activation = 'relu'
        img_rows, img_cols, img_colors = IMAGE_SIZE, IMAGE_SIZE, 1
        super(MNISTClassifier, self).__init__(name='')
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
        self.activation = layers.Activation('softmax', name='y_pred')

    def call(self, input_tensor, training=False, mask=None):
        x = self.conv1(input_tensor)
        x = self.conv2(x)
        x = self.max_pool(x)

        x = self.dropout1(x, training=training)
        x = self.flatten(x)
        x = self.fc1(x)

        x = self.dropout2(x, training=training)
        x = self.fc2(x)
        x = self.activation(x)

        return x


class MNISTTrainer:

    BATCH_SIZE = 128
    MAX_EPOCHS = 5
    LEARNING_RATE = 0.1
    LR_DECAY = 1e-6
    LR_DROP = 20

    def __init__(self, model, *args, **kwargs):
        self.model = model
        self.train_images, self.train_labels, self.test_images, \
        self.test_labels = self.initialize_dataset()
        self.original_train_labels = self.train_labels
        self.original_test_labels = self.test_labels
        self.reduce_lr = keras.callbacks.LearningRateScheduler(
            self.lr_scheduler)

    def initialize_dataset(self):
        return GANTrainer.load_data()

    def lr_scheduler(self, epoch):
        return self.LEARNING_RATE * (0.5 ** (epoch // self.LR_DROP))

    def train(self, loss, epochs=MAX_EPOCHS):
        self.model.compile(loss=loss,
                           optimizer=keras.optimizers.Adadelta(),
                           metrics=[keras.metrics.CategoricalAccuracy()])

        history = self.model.fit(self.train_images, self.train_labels,
                                 batch_size=self.BATCH_SIZE,
                                 epochs=epochs,
                                 verbose=1,
                                 validation_data=(self.test_images,
                                                  self.test_labels),
                                 callbacks=[self.reduce_lr])
        return history

    def convert_labels_to_soft_scores(self,
                                      train_soft_scores,
                                      test_soft_scores):
        self.original_train_labels = self.train_labels
        self.original_test_labels = self.test_labels
        self.train_labels = train_soft_scores
        self.test_labels = test_soft_scores

