import os

import numpy as np
import tensorflow as tf
from keras import backend as K
from scipy.optimize import fmin_l_bfgs_b
from skimage import io
from tqdm import tqdm as progress_bar

from src.utils import read_image_as_tensor, tensor_to_image


def feature_reconstruction_loss(base_image: tf.Tensor, generated_image: tf.Tensor) -> float:
    """
    Compute reconstruction loss measuring how similar generated image is to anchor image

    :param base_image: image with content representation
    :param generated_image: image generated during style transfer

    :return: values with measure of content similarity between images
    """
    return tf.reduce_sum(tf.square(generated_image - base_image))


def smoothness_loss(image: tf.Tensor) -> float:
    """
    Reduces variation between neighboring pixels required for a smooth image

    :param image: image represented as tensor of pixel intensities

    :returns: values measuring the overall smoothness of the image
    """
    return tf.reduce_sum(
        tf.square(image[:, :-1, :-1, :] - image[:, 1:, :-1, :])
        + tf.square(image[:, :-1, :-1, :] - image[:, :-1, 1:, :])
    )


def gram_matrix(matrix: tf.Tensor) -> tf.Tensor:
    """
    Compute gramian for given matrix

    :param matrix: 2D tensor with matrix values

    :return: Gramian for given tensor
    """
    values = tf.keras.backend.batch_flatten(tf.transpose(matrix, perm=(2, 0, 1)))
    return tf.tensordot(values, tf.transpose(values), axes=1)


def style_reconstruction_loss(base_image, generated_image) -> float:
    """
    Compute reconstruction loss, function outputs small values when image
    is similar to used style image in terms of linearly independent features

    :param base_image:  image with content representation
    :param generated_image: image generated during style transfer

    :returns: value with a measure of how much given image fit to given style
    """
    height = int(base_image.shape[0])
    width = int(base_image.shape[1])
    pixel_weight = 1.0 / np.power(2 * height * width, 2)

    return pixel_weight * tf.reduce_sum(tf.square(gram_matrix(generated_image) - gram_matrix(base_image)))


def style_loss_for_all_layers(style_layers: tuple, style_weights: tuple, layer_to_output_mapping):
    """
    Compute style loss for all used style layers

    :param style_layers: keys of layers used as style layers from VGG model
    :param style_weights: weights of each style layer
    :param layer_to_output_mapping: mapping between layer key and output tensor

    :return: aggregated style_reconstruction_loss for all layers
    """
    style_loss = K.variable(0.0)  # we update this variable in the loop
    weight = 1.0 / float(len(style_layers))

    for index, layer in enumerate(style_layers):
        # extract features of given layer
        style_features = layer_to_output_mapping[layer]
        # from those features, extract style and output values
        style_image_features = style_features[1, :, :, :]  # 1 corresponds to style image
        output_style_features = style_features[2, :, :, :]  # 2 corresponds to generated image
        style_loss.assign_add(
            style_weights[index]
            * weight
            * style_reconstruction_loss(style_image_features, output_style_features)
        )

    return style_loss


def style_transfer(
    content_image_path,
    style_image_path,
    outputs_dir,
    n_epochs: int,
    content_weight: float = 3e-2,
    style_weights: tuple = (20000, 500, 12, 1, 1),
    smoothness_weight: float = 5e-2,
    content_layer: str = "block4_conv2",
    style_layers: tuple = ("block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1", "block5_conv1"),
    save_frequency: int = None,
):
    width, height = tf.keras.preprocessing.image.load_img(content_image_path).size
    save_frequency = save_frequency or n_epochs

    content_image = K.variable(read_image_as_tensor(content_image_path, (height, width)))
    style_image = K.variable(read_image_as_tensor(style_image_path, (height, width)))
    generated_image = K.placeholder((1, height, width, 3))  # tensor placeholder for generated image

    input_as_tensor = tf.concat([content_image, style_image, generated_image], axis=0)
    model = tf.keras.applications.vgg19.VGG19(input_tensor=input_as_tensor, weights="imagenet", include_top=False)
    layer_to_output_mapping = {layer.name: layer.output for layer in model.layers}

    # Extract features from the content layer
    content_features = layer_to_output_mapping[content_layer]
    base_image_features = content_features[0, :, :, :]  # 0 corresponds to base
    combination_features = content_features[2, :, :, :]  # 2 corresponds to generated

    # Compute total loss
    content_loss_value = content_weight * feature_reconstruction_loss(base_image_features, combination_features)
    style_loss_value = style_loss_for_all_layers(style_layers, style_weights, layer_to_output_mapping)
    smoothness_loss_value = smoothness_weight * smoothness_loss(generated_image)
    total_loss = content_loss_value + style_loss_value + smoothness_loss_value

    # Compute gradients of output img with respect to total_loss
    grads = K.gradients(total_loss, generated_image)
    outputs = [total_loss] + grads
    loss_and_grads = K.function([generated_image], outputs)
    # Initialize the generated image from random noise
    x = np.random.uniform(0, 255, (1, height, width, 3)) - 128.

    # Fit over the total iterations
    for epoch in progress_bar(range(n_epochs)):
        x, min_val, info = fmin_l_bfgs_b(
            # extract loss function from tf model
            func=lambda x: loss_and_grads([x.reshape((1, height, width, 3))])[0],
            x0=x.flatten(),
            # extract gradients from tf model
            fprime=lambda x: loss_and_grads([x.reshape((1, height, width, 3))])[1].flatten().astype("float64"),
            maxfun=20,
        )

        if epoch % save_frequency == 0:
            generated_image = tensor_to_image(x.copy(), width, height)
            io.imsave(os.path.join(outputs_dir, f"generated_image_at_{epoch}_epoch.jpg"), generated_image)

    return tensor_to_image(x.copy(), width, height)
