import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras

from common import RANDOM_DIM, MNIST_CLASSES, RESULTS, SAVED_MODELS, \
    generate_checkpoint_path
from gan_network import Generator, Discriminator, GAN, get_gan


class GANTrainer:
    def __init__(self, gan: GAN):
        self.dLosses = []
        self.gLosses = []
        self.gan = gan
        self.generator: Generator = self.gan.generator
        self.discriminator: Discriminator = self.gan.discriminator
        self.X_train, _, _, _ = self.load_data()

    # Plot the loss from each batch
    def plot_loss(self):
        plt.figure(figsize=(10, 8))
        plt.plot(self.dLosses, label='Discriminitive loss')
        plt.plot(self.gLosses, label='Generative loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(RESULTS,
                                 "loss_vs_epoch.png"))

    @staticmethod
    def load_data():
        # Load MNIST data - Note that normalization is a little different
        # than we have used in the past
        (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = X_train[:, :, :, np.newaxis]
        X_test = (X_test.astype(np.float32) - 127.5) / 127.5
        X_test = X_test[:, :, :, np.newaxis]

        y_train = keras.utils.to_categorical(y_train, MNIST_CLASSES)
        y_test = keras.utils.to_categorical(y_test, MNIST_CLASSES)
        return X_train, y_train, X_test, y_test

    # Create a wall of generated MNIST images
    def plot_generated_images(self, epoch, examples=100, dim=(10, 10),
                              figsize=(10, 10)):
        noise = np.random.normal(0, 1, size=[examples, RANDOM_DIM])
        generated_images = self.generator.predict(noise)

        plt.figure(figsize=figsize)
        for i in range(generated_images.shape[0]):
            plt.subplot(dim[0], dim[1], i + 1)
            plt.imshow(generated_images[i].reshape((28, 28)),
                       interpolation='nearest', cmap='gray_r')
            plt.axis('off')
        title = f"Generated images @ epoch={epoch:03d}"
        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS,
                                 title.replace(" ", "_") + ".png"))

    def train(self, epochs=1, batch_size=128):
        batch_count = int(self.X_train.shape[0] / batch_size)
        print('Epochs:', epochs)
        print('Batch size:', batch_size)
        print('Batches per epoch:', batch_count)

        for epoch in range(1, epochs + 1):
            if epoch % 10 == 0 or epoch == epochs:
                generator_checkpoint_path = generate_checkpoint_path(
                    model_name="generator", epoch=epoch)
                discriminator_checkpoint_path = generate_checkpoint_path(
                    model_name="discriminator", epoch=epoch)
                self.generator.save_weights(generator_checkpoint_path)
                self.discriminator.save_weights(discriminator_checkpoint_path)
            print('\n', '-' * 15, 'Epoch ', epoch, '-' * 15)
            for batch in range(batch_count):
                print("\rBatch number: %d/%d" % (batch, batch_count), end="")

                # Get a random set of input noise and images
                noise = np.random.normal(0, 1, size=[batch_size, RANDOM_DIM])
                image_batch = self.X_train[
                    np.random.randint(0, self.X_train.shape[0],
                                      size=batch_size)]

                # Generate fake MNIST images
                generated_images = self.generator.predict(noise)
                X = np.concatenate([image_batch, generated_images])

                # Labels for generated and real data
                yDis = np.zeros(2 * batch_size)
                # One-sided label smoothing - this is done because of the
                # 'sigmoid' output of the discriminator
                yDis[:batch_size] = 0.9

                # Train discriminator
                self.discriminator.trainable = True
                dloss = self.discriminator.train_on_batch(X, yDis)

                # Train generator
                noise = np.random.normal(0, 1, size=[batch_size, RANDOM_DIM])
                yGen = np.ones(batch_size)
                self.discriminator.trainable = False
                gloss = self.gan.train_on_batch(noise, yGen)

            # Store loss of most recent batch from this epoch
            self.dLosses.append(dloss)
            self.gLosses.append(gloss)

            if epoch == 1 or epoch % 5 == 0:
                self.plot_generated_images(epoch=epoch)

        # Plot losses from every epoch
        self.plot_loss()


def main():
    # create a GAN
    adam = keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
    gan = get_gan()
    gan.compile(loss='binary_crossentropy', optimizer=adam)
    # train the GAN
    trainer = GANTrainer(gan)
    trainer.train(100, 128)


if __name__ == '__main__':
    main()
