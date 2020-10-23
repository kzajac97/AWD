import tensorflow as tf


def build_mnist_discriminator() -> tf.keras.Sequential:
    """
    :return: compiled discriminator model for generating hand written digits
    """
    model = tf.keras.Sequential()

    tf.keras.layers.Input(input_shape=(28, 28, 1))

    model.add(tf.keras.layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same"))
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
