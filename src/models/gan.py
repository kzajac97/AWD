import numpy as np
import tensorflow as tf
from tqdm import tqdm as progress_bar


def build_mnist_discriminator() -> tf.keras.Sequential:
    """
    :return: compiled discriminator model for generating hand written digits
    """
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same", input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    model.add(tf.keras.layers.Dropout(0.4))

    model.add(tf.keras.layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same"))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    model.add(tf.keras.layers.Dropout(0.4))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

    optimizer = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    return model


def build_mnist_generator(latent_dim: int = 100) -> tf.keras.Sequential:
    """
    :return: generator model for generating hand written digits
    """
    model = tf.keras.Sequential()
    # foundation for 7x7 image
    n_nodes = 128 * 7 * 7
    model.add(tf.keras.layers.Dense(n_nodes, input_dim=latent_dim))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    model.add(tf.keras.layers.Reshape((7, 7, 128)))
    # up sample to 14x14
    model.add(tf.keras.layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    # up sample to 28x28
    model.add(tf.keras.layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))

    model.add(tf.keras.layers.Conv2D(1, (7, 7), activation="sigmoid", padding="same"))
    return model


def build_mnist_gan(generator, discriminator):
    discriminator.trainable = False

    model = tf.keras.Sequential([generator, discriminator, ])

    adam = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss="binary_crossentropy", optimizer=adam)

    return model


def fit(
    model: tf.keras.Model,
    generator: tf.keras.Model,
    discriminator: tf.keras.Model,
    data_set,
    latent_dim: int,
    epochs: int,
    batch_size=256,
    silent: bool = False
) -> None:
    """
    Run Adversarial training of the model

    :param model: GAN model as joined generator and discriminator
    :param generator: generator model
    :param discriminator: discriminator model
    :param data_set: data set interface object generating desired data set
    :param latent_dim: dimension of latent space used to generate samples
    :param epochs: number of epochs to train for
    :param batch_size: number of data points in single batch
    :param silent: if False print progress bar
    """
    n_batches_in_epoch = int(data_set.n_data_points / batch_size)
    for epoch in progress_bar(range(epochs), disable=silent):
        for batch_index in range(n_batches_in_epoch):
            # set-up data batch
            samples = generator(data_set.latent_batch(batch_size // 2, latent_dim=latent_dim))
            x, y = data_set.batch(samples, size=batch_size)
            # train discriminator
            discriminator_loss, _ = discriminator.train_on_batch(x, y)
            # train generator
            inputs = data_set.latent_batch(size=batch_size, latent_dim=latent_dim)
            labels = np.ones([batch_size, 1])
            model.train_on_batch(inputs, labels)
