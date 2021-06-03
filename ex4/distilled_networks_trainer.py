import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.losses import CategoricalCrossentropy
from mnist_classifier import MNISTClassifierWithTemperature
from common import MNIST_CLASSES


class MNISTTrainer:

    BATCH_SIZE = 128
    MAX_EPOCHS = 12
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
        img_rows, img_cols, img_colors = 28, 28, 1
        (train_images, train_labels), (
            test_images, test_labels) = keras.datasets.mnist.load_data()
        train_images = train_images.astype('float32')
        test_images = test_images.astype('float32')
        train_images = train_images.reshape(train_images.shape[0], img_rows,
                                            img_cols, 1)
        test_images = test_images.reshape(test_images.shape[0], img_rows,
                                          img_cols, 1)
        train_images, test_images = self.normalize(train_images,
                                                   test_images)

        train_labels = keras.utils.to_categorical(train_labels, MNIST_CLASSES)
        test_labels = keras.utils.to_categorical(test_labels, MNIST_CLASSES)
        return train_images, train_labels, test_images, test_labels

    @staticmethod
    def normalize(x_train, x_test):
        x_train -= x_train.min()
        x_train /= x_train.max()
        x_test -= x_test.min()
        x_test /= x_test.max()
        return x_train, x_test

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


class BinaryClassificationTrainer(MNISTTrainer):
    def __init__(self, model, first_mnist_digit=0, second_mnist_digit=1):
        super(BinaryClassificationTrainer, self).__init__(model)
        self.first_mnist_digit = keras.utils.to_categorical(first_mnist_digit,
                                                            MNIST_CLASSES)
        self.second_mnist_digit = keras.utils.to_categorical(second_mnist_digit,
                                                             MNIST_CLASSES)



        self.train_images = self.filter_images(self.train_images,
                                               self.train_labels,
                                               self.first_mnist_digit,
                                               self.second_mnist_digit)
        self.train_labels = self.filter_and_rearrange_labels(
            self.train_labels, self.first_mnist_digit, self.second_mnist_digit)

        self.test_images = self.filter_images(self.test_images,
                                              self.test_labels,
                                              self.first_mnist_digit,
                                              self.second_mnist_digit)
        self.test_labels = self.filter_and_rearrange_labels(
            self.test_labels, self.first_mnist_digit, self.second_mnist_digit)

    @staticmethod
    def find_indices_to_keep(labels,
                             first_mnist_digit,
                             second_mnist_digit,
                             num_of_classes=MNIST_CLASSES):
        # filter out unused labels
        indices_to_keep = np.logical_or(
            (labels == first_mnist_digit).sum(axis=1) == num_of_classes,
            (labels == second_mnist_digit).sum(axis=1) == num_of_classes
        )
        return indices_to_keep

    @staticmethod
    def filter_and_rearrange_labels(labels, first_mnist_digit,
                                    second_mnist_digit,
                                    num_of_classes=MNIST_CLASSES):
        # filter out unused labels
        indices_to_keep = BinaryClassificationTrainer.find_indices_to_keep(
            labels, first_mnist_digit, second_mnist_digit, num_of_classes)
        labels = labels[indices_to_keep]
        # change the ground truth to support binary classification
        indices_of_first_digit = (labels == first_mnist_digit
                                  ).sum(axis=1) == num_of_classes
        indices_of_second_digit = (labels == second_mnist_digit
                                   ).sum(axis=1) == num_of_classes
        new_labels = np.zeros((labels.shape[0], 2))
        new_labels[indices_of_first_digit, ...] = np.tile(
            keras.utils.to_categorical(0, 2),
            (indices_of_first_digit.sum(), 1))
        new_labels[indices_of_second_digit, ...] = np.tile(
            keras.utils.to_categorical(1, 2),
            (indices_of_second_digit.sum(), 1))
        return new_labels

    @staticmethod
    def filter_images(images, labels, first_mnist_digit,
                      second_mnist_digit):
        # filter out unused labels
        indices_to_keep = BinaryClassificationTrainer.find_indices_to_keep(
            labels, first_mnist_digit, second_mnist_digit)
        return images[indices_to_keep]

    def convert_labels_to_soft_scores(self,
                                      train_soft_scores,
                                      test_soft_scores):
        self.original_train_labels = self.filter_and_rearrange_labels(
            self.train_labels,
            first_mnist_digit=keras.utils.to_categorical(0, 2),
            second_mnist_digit=keras.utils.to_categorical(1, 2),
            num_of_classes=2)
        self.original_test_labels = self.filter_and_rearrange_labels(
            self.test_labels,
            first_mnist_digit=keras.utils.to_categorical(0, 2),
            second_mnist_digit=keras.utils.to_categorical(1, 2),
            num_of_classes=2)
        self.train_labels = train_soft_scores
        self.test_labels = test_soft_scores


    def train(self, loss, epochs=MNISTTrainer.MAX_EPOCHS):
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


class Distillery:
    def __init__(self,
                 classifier=MNISTClassifierWithTemperature,
                 trainer=MNISTTrainer,
                 temperature: float = 1.0,
                 *trainer_args,
                 **trainer_kwargs):
        self.teacher = classifier(temperature)
        self.student = classifier(temperature)
        self.teacher_trainer = trainer(self.teacher,
                                       *trainer_args,
                                       **trainer_kwargs)
        self.student_trainer = trainer(self.student,
                                       *trainer_args,
                                       **trainer_kwargs)

    def train_teacher(self, loss=CategoricalCrossentropy()):
        teacher_history = self.teacher_trainer.train(loss=loss)
        self.teacher = self.teacher_trainer.model
        return teacher_history

    def train_student_according_to_teacher_soft_scores(
            self, epochs=MNISTTrainer.MAX_EPOCHS):
        train_soft_scores = self.teacher.predict(
            self.student_trainer.train_images)
        test_soft_scores = self.teacher.predict(
            self.student_trainer.test_images)
        self.student_trainer.convert_labels_to_soft_scores(train_soft_scores,
                                                           test_soft_scores)

        def custom_loss(teachers_scores, students_scores):
            nof_samples = teachers_scores.shape[1]
            return -1.0 / nof_samples * tf.reduce_sum(
                tf.math.log(students_scores) * teachers_scores)

        student_history = self.student_trainer.train(loss=custom_loss,
                                                     epochs=epochs)
        self.student = self.student_trainer.model
        return student_history

    def reset_student_temperature(self):
        self.student.reset_temperature()
        self.student_trainer.model.reset_temperature()

    def distill(self,
                loss=CategoricalCrossentropy(),
                epochs=MNISTTrainer.MAX_EPOCHS):
        teacher_history = self.train_teacher(loss)
        student_history = \
            self.train_student_according_to_teacher_soft_scores(epochs=epochs)
        return teacher_history, student_history


if __name__ == "__main__":
    distillery = Distillery(temperature=30)
    distillery.distill()
