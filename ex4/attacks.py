import os
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras

from common import MNIST_CLASSES, IMAGE_SIZE, RESULTS


NUM_CLASSES = MNIST_CLASSES


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
        adv_loss = keras.losses.categorical_crossentropy(target_label_tensor,
                                                       predicted)
    adv_grads = grad.gradient(adv_loss, input_tensor)

    # Finally, the FGSM formula is rather straight forward:
    # x`= x + epsilon * sign(loss(x,y))
    delta = tf.multiply(epsilon, tf.cast(tf.sign(adv_grads), dtype=tf.float32))
    adv_out = input_tensor - delta
    return adv_out.numpy()


def PGD_L2_targeted(model, images, labels, epsilon=3, iter_eps=0.03,
                    iterations=60, min_x=0.0, max_x=1.0):
    adv_out = images

    for iteration in range(iterations):
        # print('Iteration:', iteration)
        # Perturb the input
        adv_out = targeted_gradient_sign_method(model, adv_out, labels,
                                                epsilon=iter_eps)

        # Project the perturbation to the epsilon ball (L2 projection)
        perturbation = adv_out - images
        norm = np.sum(np.square(perturbation), axis=(1, 2, 3), keepdims=True)
        norm = np.sqrt(np.maximum(10e-12, norm))
        factor = np.minimum(1, np.divide(epsilon, norm))
        adv_out = np.clip(images + perturbation * factor, min_x, max_x)

    return adv_out


def PGD_L2_non_targeted(model, images, labels, epsilon=3, iter_eps=0.03,
                        iterations=60, min_x=0.0, max_x=1.0):
    adv_out = images

    for iteration in range(iterations):
        # print('Iteration:', iteration)
        # Perturb the input
        adv_out = fast_gradient_sign_method(model, adv_out, labels,
                                            epsilon=iter_eps)

        # Project the perturbation to the epsilon ball (L2 projection)
        perturbation = adv_out - images
        norm = np.sum(np.square(perturbation), axis=(1, 2, 3), keepdims=True)
        norm = np.sqrt(np.maximum(10e-12, norm))
        factor = np.minimum(1, np.divide(epsilon, norm))
        adv_out = np.clip(images + perturbation * factor, min_x, max_x)

    return adv_out


class AttackBenchmark:
    """Apply and evaluate attacks."""

    ADVERSARIAL_SAMPLE_SIZE = 1000

    def __init__(self, model_trainer, attack):
        self.trainer = model_trainer
        self.model = self.trainer.model
        self.attack = attack
        # self.is_targeted_attack = is_targeted_attack
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

    def apply_attack(self, images, labels):
        adv_images = self.attack(self.model,
                                 images,
                                 labels)
        return adv_images

    def clean_data_accuracy(self):
        score = self.model.evaluate(self.clean_data['images'],
                                    self.clean_data['labels'],
                                    verbose=0)
        return score[1]

    def pick_sample_indices_at_random(self):
        """Return the indices of the hold-out set."""
        indices = tf.range(tf.shape(self.trainer.test_images)[0])
        random_indices = tf.random.shuffle(indices)[
                         :self.ADVERSARIAL_SAMPLE_SIZE]
        return random_indices

    def choose_random_samples_from_test_set(self):
        indices = self.pick_sample_indices_at_random()
        x_0 = self.trainer.test_images[indices]
        labels = self.trainer.test_labels[indices]
        return indices, x_0, labels


def plot_adversarial_images(adv_images, old_labels, model, title=''):
    indices_of_adv_examples = random.choices(range(adv_images.shape[0]),
                                             k=16)
    plt.subplots(4, 4)
    fig = plt.gcf()
    fig.set_size_inches(10, 10)
    plt.suptitle(title)
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
                             title.replace(" ", "_") + ".png"))
    plt.close()
