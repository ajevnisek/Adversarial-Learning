import os
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy
from common import RESULTS, MNIST_CLASSES, toCyan, toMagenta, toGreen
from mnist_classifier import MNISTClassifierWithTemperature, \
    BinaryClassifierWithTemperature
from distilled_networks_trainer import Distillery, MNISTTrainer, \
    BinaryClassificationTrainer
from attacks import AttackBenchmark, fast_gradient_sign_method, \
    targeted_gradient_sign_method, PGD_L2_non_targeted, PGD_L2_targeted, \
    plot_adversarial_images


ATTACK_NAME_TO_ATTACK = {'FGSM attack': fast_gradient_sign_method,
                         'Targeted FGSM attack': targeted_gradient_sign_method,
                         'Non-targeted PGD attack': PGD_L2_non_targeted,
                         'Targeted PGD attack': PGD_L2_targeted,
                         }

def benchmark_attack_on_training_scheme(attack,
                                        training_scheme: MNISTTrainer,
                                        is_targeted_attack: bool = False,
                                        title: str = '',
                                        num_of_classes=MNIST_CLASSES):
    benchmark = AttackBenchmark(training_scheme, attack)
    indices, images, labels = \
        benchmark.choose_random_samples_from_test_set()
    if is_targeted_attack:
        target = (np.argmax(labels, axis=1) +
                  np.random.randint(
                      1, num_of_classes, size=(labels.shape[0]))) % \
                 num_of_classes
        target_labels = keras.utils.to_categorical(target, num_of_classes)
        adversarial_images = benchmark.apply_attack(images, target_labels)

    else:
        target_labels = None
        adversarial_images = benchmark.apply_attack(images, labels)

    loss, accuracy, mean_perturbation = benchmark.test_attack(
        adversarial_images, images, labels, targeted=is_targeted_attack,
        target_labels=target_labels)

    plot_adversarial_images(adversarial_images, labels, training_scheme.model,
                            title=title)
    return loss, accuracy, mean_perturbation


def evaluate_distillation_performance_for_mnist_classifier():
    # train a student according to a teacher trained with temperature
    distillery = Distillery()
    distillery.distill(epochs=5)
    # reset the student's temperature to 1 at inference time
    distillery.reset_student_temperature()
    # train a MNIST classifier without distillation
    simple_classifier = MNISTClassifierWithTemperature(temperature=1)
    classical_trainer = MNISTTrainer(simple_classifier)
    classical_trainer.initialize_dataset()
    classical_trainer.train(loss=CategoricalCrossentropy())
    # define the two learning schemes: classical and distilled
    scheme_name_to_classification_scheme = {
        'Simple Classifier': classical_trainer,
        'Distilled Student': distillery.student_trainer}
    # loop over the attacks and report performance
    for attack_name in ATTACK_NAME_TO_ATTACK:
        for scheme_name in scheme_name_to_classification_scheme:
            attack = ATTACK_NAME_TO_ATTACK[attack_name]
            scheme = scheme_name_to_classification_scheme[scheme_name]
            is_targeted_attack = attack_name.startswith("Targeted")
            title = f'{attack_name} on {scheme_name}'
            color_function = toCyan if 'Simple' in scheme_name else toMagenta
            print(color_function(title))
            benchmark_attack_on_training_scheme(
                attack, scheme, is_targeted_attack=is_targeted_attack,
                title=title)
            print('=' * 20)


def get_logits_and_soft_scores_for_model(model_trainer, gt_labels):
    model = model_trainer.model
    images = model_trainer.test_images
    class_zero_indices = (gt_labels == keras.utils.to_categorical(0, 2)).sum(
        axis=1) == 2
    class_zero_images = images[class_zero_indices]

    class_zero_softmax_scores = model.predict(class_zero_images)
    _ = model.call(class_zero_images)
    class_zero_logits = model.logits
    return class_zero_softmax_scores, class_zero_logits


def plot_diagram(softmax_scores,
                 logits,
                 path,
                 name='Simple Classifier'):
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.scatter(logits[:, 0],
                logits[:, 1],
                c='b', marker='s', label='class 0')
    lim_max = max([np.array(logits[:, 0]).max(), np.array(logits[:, 1]).max()])
    lim_min = max([np.array(logits[:, 0]).min(), np.array(logits[:, 1]).min()])
    ax1.plot(np.arange(lim_min, lim_max, 0.1),
             np.arange(lim_min, lim_max, 0.1),
             '-k')
    ax1.set_title(f'{name} logits')
    ax1.set_xlabel(f'logit 0')
    ax1.set_ylabel(f'logit 1')
    ax2 = fig.add_subplot(122)
    ax2.scatter(softmax_scores[:, 0],
                softmax_scores[:, 1],
                c='r', marker='o', label='class 1')
    lim_max = max([np.array(softmax_scores[:, 0]).max(),
                   np.array(softmax_scores[:, 1]).max()])
    lim_min = max([np.array(softmax_scores[:, 0]).min(),
                   np.array(softmax_scores[:, 1]).min()])
    ax2.plot(np.arange(lim_min, lim_max, 0.1),
             np.arange(lim_min, lim_max, 0.1),
             '-k')
    ax2.set_title(f'{name} Softmax scores')
    ax2.set_xlabel(f'softmax 0')
    ax2.set_ylabel(f'softmax 1')
    fig.set_size_inches(8, 8)
    fig.suptitle(f'{name} network activations')
    plt.savefig(path)


def evaluate_distillation_performance_for_binary_classifier():
    # randomly pick two classes
    first_mnist_digit, second_mnist_digit = random.choices(range(
        MNIST_CLASSES), k=2)
    # train a student according to a teacher trained with temperature
    trainer_kwargs = {'first_mnist_digit': first_mnist_digit,
                      'second_mnist_digit': second_mnist_digit}
    distillery = Distillery(classifier=BinaryClassifierWithTemperature,
                            trainer=BinaryClassificationTrainer,
                            temperature=30.0,
                            **trainer_kwargs)
    distillery.distill(loss=BinaryCrossentropy(), epochs=5)
    # reset the student's temperature to 1 at inference time
    distillery.reset_student_temperature()
    # loop over the attacks and report performance
    for attack_name in ATTACK_NAME_TO_ATTACK:
        attack = ATTACK_NAME_TO_ATTACK[attack_name]
        scheme = distillery.student_trainer
        is_targeted_attack = attack_name.startswith("Targeted")
        title = f'{attack_name} on Distilled Binary Classification Student'
        print(toGreen(title))
        benchmark_attack_on_training_scheme(
            attack, scheme, is_targeted_attack=is_targeted_attack,
            title=title, num_of_classes=2)
        print('=' * 20)
    # train a simple binary classifier
    simple_classifier = BinaryClassifierWithTemperature()

    bc_trainer = BinaryClassificationTrainer(simple_classifier,
                                             first_mnist_digit,
                                             second_mnist_digit)
    bc_trainer.train(loss=BinaryCrossentropy())
    # print all diagrams:
    s0, l0 = get_logits_and_soft_scores_for_model(
        bc_trainer, gt_labels=bc_trainer.test_labels)
    plot_diagram(s0,
                 l0,
                 path=os.path.join(
                     RESULTS,
                     "simple_binary_classifier_performance.png"),)
    s0, l0 = get_logits_and_soft_scores_for_model(
        distillery.student_trainer,
        gt_labels=distillery.student_trainer.original_test_labels)
    plot_diagram(s0,
                 l0,
                 path=os.path.join(
                     RESULTS,
                     "distilled_binary_classifier_performance.png"),
                 name='Distilled Classifier')


def main():
    # solution to items 1 - 5:
    evaluate_distillation_performance_for_mnist_classifier()
    # solution to items 6 - 8:
    evaluate_distillation_performance_for_binary_classifier()


if __name__ == "__main__":
    main()
