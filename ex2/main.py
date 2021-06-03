import os
import random
import termcolor

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers


IMAGE_SIZE = 28

NUM_CLASSES = 10
BATCH_SIZE = 128
MAX_EPOCHS = 12

LEARNING_RATE = 0.1
LR_DROP = 20

ADVERSARIAL_SAMPLE_SIZE = 1000
RESULTS = "results"


def to_blue(content): return termcolor.colored(content, "blue", attrs=["bold"])


def build_mnist_model():
    """ Build a simple MNIST classification CNN
        The network takes ~3 minutes to train on a normal laptop and reaches
        roughly 97% of accuracy.
        Model structure: Conv, Conv, Max pooling, Dropout, Dense, Dense
    """
    activation = 'relu'
    # input image dimensions
    img_rows, img_cols, img_colors = IMAGE_SIZE, IMAGE_SIZE, 1

    model = keras.Sequential()
    model.add(layers.Conv2D(8, kernel_size=(3, 3),
                            input_shape=(img_rows, img_cols, img_colors),
                            activation=activation))
    model.add(layers.Conv2D(8, (3, 3), activation=activation))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation=activation))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(NUM_CLASSES))
    model.add(layers.Activation('softmax', name='y_pred'))

    return model


def fast_gradient_sign_method(model, images, labels, epsilon=0.3):
    """Fast Gradient Sign Method implementation - perturb all input features by
    an epsilon sized step in the direction of loss gradient."""
    # The GradientTape is the context at which we can explicitly ask for
    # gradient calculation.
    # We define the relevant tensors inside that context, and ask for the
    # gradient calculation outside of it.
    with tf.GradientTape() as grad:
      true_label_tensor = tf.Variable(labels, dtype=tf.float32)
      input_tensor = tf.Variable(images, dtype=tf.float32)
      predicted = model(input_tensor)
      adv_loss = keras.losses.categorical_crossentropy(true_label_tensor,
                                                       predicted)
    adv_grads = grad.gradient(adv_loss, input_tensor)

    # Finally, the FGSM formula is rather straight forward:
    # x`= x + epsilon * sign(loss(x,y))
    delta = tf.multiply(epsilon, tf.cast(tf.sign(adv_grads), dtype=tf.float32))
    adv_out = input_tensor + delta
    return adv_out.numpy()


def targeted_gradient_sign_method(model, images, target, epsilon=0.3):
    """Targeted Gradient Sign Method implementation - A targeted variant of the
    FGSM attack here we minimize the loss with respect to the target class,
    as opposed to maximizing the loss with respect to the source class"""
    # The GradientTape is the context at which we can explicitly ask for
    # gradient calculation.
    # We define the relevant tensors inside that context, and ask for the
    # gradient calculation outside of it.
    with tf.GradientTape() as grad:
      target_label_tensor = tf.Variable(target, dtype=tf.float32)
      input_tensor = tf.Variable(images, dtype=tf.float32)
      predicted = model(input_tensor)
      adv_loss = keras.losses.categorical_crossentropy(target_label_tensor, predicted)
    adv_grads = grad.gradient(adv_loss, input_tensor)

    # Finally, the FGSM formula is rather straight forward:
    # x`= x + epsilon * sign(loss(x,y))
    delta = tf.multiply(epsilon, tf.cast(tf.sign(adv_grads), dtype=tf.float32))
    adv_out = input_tensor - delta
    return adv_out.numpy()


class Trainer:
    """MNIST classifier trainer."""
    def __init__(self, model: keras.Sequential):
        self.model = model
        self.reduce_lr = keras.callbacks.LearningRateScheduler(
            self.lr_scheduler)

    @staticmethod
    def normalize(x):
        x -= x.min()
        x /= x.max()
        return x

    @staticmethod
    def lr_scheduler(epoch):
        return LEARNING_RATE * (0.5 ** (epoch // LR_DROP))

    @staticmethod
    def plot_learning_statistics(loss, categorical_accuracy, path):
        fig, axs = plt.subplots(1, 2)
        axs[0].plot(range(1, 1 + len(loss)), loss)
        axs[0].set_title('Loss')
        axs[0].set_xlabel('epoch')
        axs[0].set_ylabel('Loss')
        axs[1].plot(range(1, 1 + len(categorical_accuracy)),
                    categorical_accuracy)
        axs[1].set_xlabel('epoch')
        axs[1].set_ylabel('Categorical Accuracy')
        axs[1].set_title('Categorical Accuracy')
        fig.set_size_inches(10, 10)
        plt.savefig(path)
        plt.close()

    def load_and_prepare_dataset(self):
        img_rows, img_cols, img_colors = IMAGE_SIZE, IMAGE_SIZE, 1
        (train_images, train_labels), (
            test_images, test_labels) = keras.datasets.mnist.load_data()
        train_images = train_images.astype('float32')
        test_images = test_images.astype('float32')
        train_images = train_images.reshape(train_images.shape[0], img_rows,
                                            img_cols, 1)
        test_images = test_images.reshape(test_images.shape[0], img_rows,
                                          img_cols,
                                          1)
        self.train_images = self.normalize(train_images)
        self.test_images = self.normalize(test_images)

        self.train_labels = keras.utils.to_categorical(train_labels,
                                                       NUM_CLASSES)
        self.test_labels = keras.utils.to_categorical(test_labels, NUM_CLASSES)

    def single_epoch_train(self):
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=keras.optimizers.Adadelta(),
                           metrics=[keras.metrics.CategoricalAccuracy()])
        history = self.model.fit(self.train_images, self.train_labels,
                                 batch_size=BATCH_SIZE,
                                 epochs=1,
                                 verbose=1,
                                 validation_data=(self.test_images,
                                                  self.test_labels),
                                 callbacks=[self.reduce_lr])
        return history

    def train(self, path=os.path.join(RESULTS, "basic_train_performance.png")):
        loss, categorical_accuracy = [], []
        for _ in range(MAX_EPOCHS):
            history = self.single_epoch_train()
            loss.append(history.history['loss'])
            categorical_accuracy.append(history.history['categorical_accuracy'])
        self.plot_learning_statistics(loss, categorical_accuracy, path)


class AttackBenchmark:
    """Applies and evaluate attacks, rearranges trainer dataset at ease."""
    def __init__(self, model_trainer, attack):
        self.trainer = model_trainer
        self.model = self.trainer.model
        self.attack = attack
        self.clean_data = {'images': np.copy(self.trainer.test_images),
                           'labels': np.copy(self.trainer.test_labels)}

    def test_attack(self, adv_images, orig_images, true_labels,
                    target_labels=None, targeted=False):
        """Evaluate the success of an attack."""
        score = self.model.evaluate(adv_images, true_labels, verbose=0)
        print('Test loss: {:.2f}'.format(score[0]))
        print('Successfully moved out of source class: {:.2f}'.format(
            1 - score[1]))

        if targeted:
            score = self.model.evaluate(adv_images, target_labels, verbose=0)
            print('Test loss: {:.2f}'.format(score[0]))
            print('Successfully perturbed to target class: {:.2f}'.format(
                score[1]))

        dist = np.mean(np.sqrt(
            np.mean(np.square(adv_images - orig_images), axis=(1, 2, 3))))
        print('Mean perturbation distance: {:.2f}'.format(dist))
        return score[0], 1 - score[1], dist

    def apply_attack(self, x_0, labels):
        adv_images = self.attack(self.model,
                                 x_0,
                                 labels,
                                 epsilon=0.3)
        return adv_images

    def create_a_holdout_from_trainer_test_set(self):
        """Return the indices of the hold-out set."""
        indices = tf.range(tf.shape(self.trainer.test_images)[0])
        random_indices = tf.random.shuffle(indices)[:ADVERSARIAL_SAMPLE_SIZE]
        return random_indices

    def choose_samples_to_move(self):
        indices = self.create_a_holdout_from_trainer_test_set()
        x_0 = self.trainer.test_images[indices]
        labels = self.trainer.test_labels[indices]
        return indices, x_0, labels

    def move_samples_from_train_to_test_set(self, indices, x_0, labels):
        self.trainer.train_images = tf.concat(
            [self.trainer.train_images, x_0], 0)
        self.trainer.train_labels = tf.concat(
            [self.trainer.train_labels, labels], 0)

        # filter out images
        mask = tf.ones((self.trainer.test_images.shape))
        mask = np.array(mask)
        mask[indices, ...] = 0
        nof_images_left_in_dataset = int(mask.sum() / IMAGE_SIZE / IMAGE_SIZE)
        self.trainer.test_images = tf.boolean_mask(self.trainer.test_images,
                                                   mask)
        self.trainer.test_images = np.array(self.trainer.test_images)
        self.trainer.test_images = self.trainer.test_images.reshape(
            nof_images_left_in_dataset, IMAGE_SIZE, IMAGE_SIZE, 1)
        # filter out labels
        mask = tf.ones((self.trainer.test_labels.shape))
        mask = np.array(mask)
        mask[indices, ...] = 0
        self.trainer.test_labels = tf.boolean_mask(self.trainer.test_labels,
                                                   mask)
        self.trainer.test_labels = np.array(self.trainer.test_labels)
        self.trainer.test_labels = self.trainer.test_labels.reshape(
            nof_images_left_in_dataset, NUM_CLASSES)

    def clean_data_accuracy(self):
        score = self.model.evaluate(self.clean_data['images'],
                                    self.clean_data['labels'],
                                    verbose=0)
        return score[1]


def plot_adversarial_images(adv_images, old_labels, model, iteration):
    indices_of_adv_examples = random.choices(range(adv_images.shape[0]),
                                             k=16)
    plt.subplots(4, 4)
    fig = plt.gcf()
    fig.set_size_inches(10, 10)
    plt.suptitle(f"Adversarial Examples @ iteration {iteration}")
    for i in range(4):
        for j in range(4):
            plt.subplot(4, 4, 4 * i + j + 1)
            index = indices_of_adv_examples[4 * i + j]
            plt.imshow(adv_images[index])
            old_label = np.argmax(old_labels[index], axis=0)
            new_label = np.argmax(
                model(
                    adv_images[index].reshape(1, IMAGE_SIZE, IMAGE_SIZE, 1),
                    training=False)
            )
            plt.title(f"{old_label}->{new_label}")
            plt.xticks([])
            plt.yticks([])

    plt.savefig(os.path.join(RESULTS,
                             f"iteration_{iteration}_adversarial_examples.png"))
    plt.close()


def main():
    # implement a base classifier for MNIST.
    model = build_mnist_model()
    # train the classifier.
    trainer = Trainer(model)
    trainer.load_and_prepare_dataset()
    trainer.train()
    # use create an attack benchmark to create 1000 adversarial samples.
    attack_bm = AttackBenchmark(trainer, fast_gradient_sign_method)
    indices, x_0, x_0_labels = attack_bm.choose_samples_to_move()
    x_0_adv_images = attack_bm.apply_attack(x_0, x_0_labels)
    # test the attack using these 1000 adversarial samples.
    loss, accuracy_x0, mean_perturbation = attack_bm.test_attack(
        x_0_adv_images, x_0, x_0_labels, targeted=False)
    # move the samples from one dataset to another.
    attack_bm.move_samples_from_train_to_test_set(
        indices, x_0_adv_images, x_0_labels)
    # train a single epoch.
    attack_bm.trainer.single_epoch_train()
    # log adversarial examples in a graph.
    plot_adversarial_images(x_0_adv_images, x_0_labels,
                            attack_bm.trainer.model, iteration=0)
    message = f"Iteration ({0}) \t " \
              f"Clean data accuracy: " \
              f"{attack_bm.clean_data_accuracy() * 100:.2f} \t" \
              f"Attack success rate for x_0': {accuracy_x0 * 100:.2f} \t" \
              f"Mean L2 perturbation distance for x_i: " \
              f"{mean_perturbation:.4f} \t"
    print(to_blue(message))
    for iteration in range(1, 1 + 5):
        # choose 1000 samples at random and apply an attack on them.
        indices, x_i, labels = attack_bm.choose_samples_to_move()
        adv_images = attack_bm.apply_attack(x_i, labels)
        # test attack on the new samples.
        loss, accuracy_x0, mean_perturbation = attack_bm.test_attack(
            adv_images, x_i, labels, targeted=False)
        # test attack on the first samples we chose.
        loss, accuracy, mean_perturbation = attack_bm.test_attack(
            x_0_adv_images,  x_0, x_0_labels, targeted=False)
        # move samples from the test set to the train set.
        attack_bm.move_samples_from_train_to_test_set(
            indices, adv_images, labels)
        # train a single epoch.
        attack_bm.trainer.single_epoch_train()
        # report results to screen.
        message = f"Iteration ({iteration}) \t " \
                  f"Clean data accuracy: " \
                  f"{attack_bm.clean_data_accuracy() * 100:.2f} \t" \
                  f"Attack success rate for x_0': {accuracy_x0 * 100:.2f} \t" \
                  f"Attack success rate for x_i: {accuracy * 100:.2f} \t" \
                  f"Mean L2 perturbation distance for x_i: " \
                  f"{mean_perturbation:.4f} \t"
        print(to_blue(message))
        # plot adversarial examples in each iteration.
        plot_adversarial_images(adv_images, labels,
                                attack_bm.trainer.model, iteration=iteration)


if __name__ == "__main__":
    main()
