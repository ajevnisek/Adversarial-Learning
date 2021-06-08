import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from common import RANDOM_DIM, IMAGE_SIZE



class Generator(tf.keras.Model):
    """Generate MNIST digits from random noise.

    Compile me with binary cross-entropy and Adam optimizer.
    >>>> adam = keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
    >>>> generator = Generator()
    >>>> generator.compile(loss='binary_crossentropy', optimizer=adam)

    """
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = layers.Dense(7 * 7 * 128, input_dim=RANDOM_DIM)
        self.leaky_relu1 = layers.LeakyReLU(0.2)
        self.reshape = layers.Reshape((7, 7, 128))
        self.up_sample1 = layers.UpSampling2D(size=(2, 2))
        self.conv1 = layers.Conv2D(64, kernel_size=(5, 5), padding='same')
        self.leaky_relu2 = layers.LeakyReLU(0.2)
        self.up_sample2 = layers.UpSampling2D(size=(2, 2))
        self.conv2 = layers.Conv2D(1, kernel_size=(5, 5), padding='same',
                                   activation='tanh')

    def call(self, input_tensor, training=False, mask=None):
        x = self.fc1(input_tensor)
        x = self.leaky_relu1(x)
        x = self.reshape(x)
        x = self.up_sample1(x)
        x = self.conv1(x)
        x = self.leaky_relu2(x)
        x = self.up_sample2(x)
        x = self.conv2(x)
        return x


class Discriminator(tf.keras.Model):
    """Discriminates between original and synthetic images.

    Compile me with binary cross-entropy and Adam optimizer.
    >>>> adam = keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
    >>>> discriminator = Discriminator()
    >>>> discriminator.compile(loss='binary_crossentropy', optimizer=adam)

    """
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = layers.Conv2D(64, kernel_size=(5, 5), strides=(2, 2),
                                   padding='same',
                                   input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1))
        self.leaky_relu1 = layers.LeakyReLU(0.2)
        self.dropout1 = layers.Dropout(0.3)
        self.conv2 = layers.Conv2D(128, kernel_size=(5, 5), strides=(2, 2),
                                   padding='same')
        self.leaky_relu2 = layers.LeakyReLU(0.2)
        self.dropout2 = layers.Dropout(0.3)
        self.flatten = layers.Flatten()
        self.fc = layers.Dense(1, activation='sigmoid')

    def call(self, input_tensor, training=False, mask=None):
        x = self.conv1(input_tensor)
        x = self.leaky_relu1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.leaky_relu2(x)
        x = self.dropout2(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class GAN(tf.keras.Model):
    """Create images using adversarially to a discriminator.

        Compile me with binary cross-entropy and Adam optimizer.
        >>>> adam = keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
        >>>> gan = GAN(adam)
        >>>> gan.compile(loss='binary_crossentropy', optimizer=adam)

        """
    def __init__(self, gan_input, gan_output, discriminator, generator):
        super(GAN, self).__init__(inputs=gan_input, outputs=gan_output)
        self.generator = generator
        self.discriminator = discriminator


def get_gan():
    adam = keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
    discriminator = Discriminator()
    generator = Generator()
    discriminator.compile(loss='binary_crossentropy', optimizer=adam)
    generator.compile(loss='binary_crossentropy', optimizer=adam)
    discriminator.trainable = False
    gan_input = layers.Input(shape=(RANDOM_DIM,))
    x = generator(gan_input)
    gan_output = discriminator(x)
    return GAN(gan_input, gan_output, discriminator, generator)
