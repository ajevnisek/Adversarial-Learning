from tqdm import tqdm
from tensorflow import keras

from common import find_latest_checkpoint, toCyan, toRed, toGreen, toYellow, \
    toMagenta, IMAGE_SIZE

from gan_network import get_gan
from gan_trainer import GANTrainer
from mnist_classifier import MNISTTrainer
from mnist_classifier import MNISTClassifier
from latent_space_searcher import Searcher, ConvergedSignal
from attacks import benchmark_attack_on_training_scheme, \
    fast_gradient_sign_method, targeted_gradient_sign_method, \
    PGD_L2_non_targeted, PGD_L2_targeted


RETRAIN_GAN = False


def detector_based_defense_mechanism(generator, epsilon, x, classifier):
    searcher = Searcher(generator, epsilon=epsilon)
    search_result = searcher.search_close_z(x)

    if search_result.converged_signal is ConvergedSignal.NOT_CONVERGED:
        return 'adversarial'
    else:
        return classifier.predict(generator.predict(search_result.final_z))


def generate_defense_false_positive_statistics(attack_name,
                                               pristine_images,
                                               true_labels, generator,
                                               mnist_classifier):
    false_positive = 0
    nof_samples = len(pristine_images)
    tqdm_bar = tqdm(range(nof_samples))
    for i in tqdm_bar:
        image = pristine_images[i].reshape(
            1, IMAGE_SIZE, IMAGE_SIZE, 1)
        classification = detector_based_defense_mechanism(generator, 15,
                                                          image,
                                                          mnist_classifier)
        true_label = true_labels[i].argmax()
        if classification == 'adversarial':
            false_positive += 1
        elif classification.argmax() == true_label.argmax():
            false_positive += 0
        else:
            false_positive += 1
    message = f"For {attack_name} the FPR is {false_positive / 100}"
    print(toMagenta(message))


def generate_defense_statistics(attack_name, adversarial_images, true_labels,
                                generator, mnist_classifier):
    correctly_classified = 0
    correctly_identified = 0
    wrongly_classified = 0
    nof_samples = len(adversarial_images)
    tqdm_bar = tqdm(range(nof_samples))
    for i in tqdm_bar:
        adversarial_image = adversarial_images[i].reshape(
            1, IMAGE_SIZE, IMAGE_SIZE, 1)
        classification = detector_based_defense_mechanism(generator, 15,
                                                          adversarial_image,
                                                          mnist_classifier)
        true_label = true_labels[i].argmax()
        if classification == 'adversarial':
            correctly_identified += 1
        elif classification.argmax() == true_label.argmax():
            correctly_classified += 1
        else:
            wrongly_classified += 1
    message = f"For {attack_name} the number of correctly classified to the " \
              f"original class label is: {correctly_classified}"
    print(toGreen(message))

    message = f"For {attack_name} the number Correctly identified by our " \
              f"detector is: {correctly_identified}."
    print(toYellow(message))

    message = f"For {attack_name} the number of wrongly classified " \
              f"is: {wrongly_classified}"
    print(toRed(message))


def main():
    # (1): train a GAN to generate MNIST digits:
    adam = keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
    gan = get_gan()
    gan.compile(loss='binary_crossentropy', optimizer=adam)
    if RETRAIN_GAN:
        gan_trainer = GANTrainer(gan)
        gan_trainer.train(100, 128)
    else:
        last_checkpoint = find_latest_checkpoint(model_name='generator')
        gan.generator.load_weights(last_checkpoint)
        last_checkpoint = find_latest_checkpoint(model_name='discriminator')
        gan.discriminator.load_weights(last_checkpoint)
    # (2): train a MNIST classifier using the same input normalization as the
    # GAN's
    mnist_classifier = MNISTClassifier()
    mnist_trainer = MNISTTrainer(mnist_classifier)
    mnist_trainer.train(loss=keras.losses.CategoricalCrossentropy())
    # (4): create 100 images from each attack:
    attack_name_to_attack = {'FGSM attack': fast_gradient_sign_method,
                             'Targeted FGSM attack':
                                 targeted_gradient_sign_method,
                             'Non-targeted PGD attack': PGD_L2_non_targeted,
                             'Targeted PGD attack': PGD_L2_targeted,
                             }

    # loop over the attacks and report performance
    attack_name_to_adversarial_images = {}
    attack_name_to_true_labels = {}
    attack_name_to_target_labels = {}

    for attack_name in attack_name_to_attack:
        attack = attack_name_to_attack[attack_name]
        scheme = mnist_trainer
        is_targeted_attack = attack_name.startswith("Targeted")
        title = f'{attack_name} on MNIST Classifier'
        color_function = toCyan
        print(color_function(title))
        _, _, _, adversarial_images, true_labels, target_labels = \
            benchmark_attack_on_training_scheme(
            attack, scheme, is_targeted_attack=is_targeted_attack,
            title=title)
        attack_name_to_adversarial_images[attack_name] = adversarial_images
        attack_name_to_true_labels[attack_name] = true_labels
        attack_name_to_target_labels[attack_name] = target_labels
        print('=' * 20)
    # (5) test the defense on the attacks:
    for attack_name in attack_name_to_adversarial_images.keys():
        generate_defense_statistics(
            attack_name=attack_name,
            adversarial_images=attack_name_to_adversarial_images[attack_name],
            true_labels=attack_name_to_true_labels[attack_name],
            generator=gan.generator,
            mnist_classifier=mnist_classifier)
    # (6) test the defense on the VALID inputs:
    for attack_name in attack_name_to_adversarial_images.keys():
        generate_defense_false_positive_statistics(
            attack_name=attack_name,
            pristine_images=mnist_trainer.test_images[:100, ...],
            true_labels=mnist_trainer.test_labels[:100, ...],
            generator=gan.generator,
            mnist_classifier=mnist_classifier)


if __name__ == '__main__':
    main()
