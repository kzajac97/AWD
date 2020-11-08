import numpy as np
import tensorflow as tf
from tqdm import tqdm as progress_bar

from src.datasets import DataSetI


class GenerativeMatchingNetwork:
    def __init__(self, latent_dim: int = 100):
        """
        :param latent_dim: dimension of latent space input to generating model
        """
        self._latent_dim = latent_dim
        self._model = tf.keras.Sequential()

        # foundation for 7x7 image
        n_nodes = 128 * 7 * 7
        self._model.add(tf.keras.layers.Dense(n_nodes, input_dim=latent_dim))
        self._model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        self._model.add(tf.keras.layers.Reshape((7, 7, 128)))
        # up sample to 14x14
        self._model.add(tf.keras.layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"))
        self._model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        # up sample to 28x28
        self._model.add(tf.keras.layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"))
        self._model.add(tf.keras.layers.LeakyReLU(alpha=0.2))

        self._model.add(tf.keras.layers.Conv2D(1, (7, 7), activation="sigmoid", padding="same"))

    def __str__(self):
        return self._model.summary()

    def __repr__(self):
        return f"{type(self)} object at {hex(id(self))} containing GMN\n"

    def compile(self, learning_rate: float = 0.0002, beta: float = 0.5, loss: str = "mse") -> None:
        """
        Compile GMN tensorflow model

        :param learning_rate: learning rate used by Adam optimizer
        :param beta: parameter used by optimizer
        :param loss: loss function used during fitting the model
        """
        self._model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate, beta_1=beta))

    def fit(self, data_set: DataSetI, n_epochs: int, batch_size: int = 256, silent: bool = False) -> None:
        """
        Fit model to generate given data set

        :param data_set: DataSetI object containing desired data set
        :param n_epochs: number of iterations run in training
        :param batch_size: size of data batch
        :param silent: if False print progress bar
        """
        n_batches_in_epoch = int(data_set.n_data_points / batch_size)

        for epoch in progress_bar(range(n_epochs), disable=silent):
            for batch_index in range(n_batches_in_epoch):
                # set-up data batch
                samples = data_set.latent_batch(batch_size, self._latent_dim)
                labels = data_set.real_batch(batch_size)

                _ = self._model.train_on_batch(samples, labels)

        return

    def generate(self, latent_batch: np.array) -> np.array:
        """Generate samples based on given latent batch"""
        return self._model.predict(latent_batch)
