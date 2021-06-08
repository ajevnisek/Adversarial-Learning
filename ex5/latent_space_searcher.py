import numpy as np
import tensorflow as tf

from enum import Enum, auto

from common import RANDOM_DIM


class ConvergedSignal(Enum):
    CONVERGED = auto()
    NOT_CONVERGED = auto()


class SearchResult:
    def __init__(self, converged_signal: ConvergedSignal, final_z, iterations,
                 l2_distance):
        self.converged_signal = converged_signal
        self.final_z = final_z
        self.iterations = iterations
        self.l2_distance = l2_distance


class Searcher:
    MAX_ITERATIONS = 50

    def __init__(self, generator, epsilon, learning_rate=1e-1):
        self.generator = generator
        self.epsilon = epsilon
        self.learning_rate = learning_rate

    def search_close_z(self, x):
        z = tf.convert_to_tensor(np.random.normal(0, 1, size=[1, RANDOM_DIM]))
        x_hat = self.generator.predict(z)
        l2_distance = tf.norm(x_hat - x)
        # l2_distances = [l2_distance.numpy()]
        iterations = 0
        while (l2_distance > self.epsilon) and (iterations <
                                                self.MAX_ITERATIONS):
            z = self.perform_gradient_step(x, z)
            l2_distance = self.compute_l2_distance(x, z)
            iterations += 1
            # l2_distances.append(l2_distance.numpy())
        # import matplotlib.pyplot as plt
        # plt.plot(l2_distances)
        # plt.show()
        if l2_distance > self.epsilon:
            return SearchResult(converged_signal=ConvergedSignal.NOT_CONVERGED,
                                final_z=z, iterations=iterations,
                                l2_distance=l2_distance)
        return SearchResult(converged_signal=ConvergedSignal.CONVERGED,
                            final_z=z, iterations=iterations,
                            l2_distance=l2_distance)

    def compute_l2_distance(self, x, z):
        x_hat = self.generator.predict(z)
        l2_distance = tf.norm(x_hat - x)
        return l2_distance

    def perform_gradient_step(self, x, z):
        z = tf.Variable(z)
        with tf.GradientTape() as tape:
            tape.watch(z)
            x_hat = self.generator(z)
            l2_distance = tf.norm(x_hat - x)
        d_l2_distance_dz = tape.gradient(l2_distance, z)
        z_new = z - self.learning_rate * d_l2_distance_dz
        return z_new


