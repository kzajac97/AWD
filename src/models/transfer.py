import numpy as np
import tensorflow as tf
from keras import backend as K
from scipy.optimize import fmin_l_bfgs_b
from tqdm import tqdm as progress_bar

from src.utils import read_image_as_tensor, tensor_to_image


def feature_reconstruction_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Reconstruction loss computed as a difference between"""
    return tf.reduce_sum(tf.square(y_pred - y_true))


def total_variation_loss(tensor):
    """
    Reduces variation between neighboring pixels
    Required for a smooth image
    """
    return tf.reduce_sum(
        tf.square(tensor[:, :-1, :-1, :] - tensor[:, 1:, :-1, :])
        + tf.square(tensor[:, :-1, :-1, :] - tensor[:, :-1, 1:, :])
    )


def gram_matrix(matrix: tf.Tensor) -> tf.Tensor:
    """
    Compute gramian for given matrix

    :param matrix: 2D tensor with matrix values

    :return: Gramian for given tensor
    """
    values = tf.keras.backend.batch_flatten(tf.transpose(matrix, perm=(2, 0, 1)))
    return tf.tensordot(values, tf.transpose(values), axes=1)


def style_reconstruction_loss(base, output):
    """
    Compute reconstruction loss, function outputs small values when image
    is similar to used style image in terms of linearly independent features
    """
    height, width = int(base.shape[0]), int(base.shape[1])
    alpha = 1.0 / float((2*height*width)**2)

    return alpha * tf.reduce_sum(tf.square(gram_matrix(output) - gram_matrix(base)))


def style_loss_for_all_layers(style_layers, style_weights, layer_to_output_mapping):
    temp_style_loss = K.variable(0.0)  # we update this variable in the loop
    weight = 1.0 / float(len(style_layers))

    for index, layer in enumerate(style_layers):
        # extract features of given layer
        style_features = layer_to_output_mapping[layer]
        # from those features, extract style and output activations
        style_image_features = style_features[1, :, :, :]  # 1 corresponds to style image
        output_style_features = style_features[2, :, :, :]  # 2 coresponds to generated image
        temp_style_loss.assign_add(
            style_weights[index]
            * weight
            * style_reconstruction_loss(style_image_features, output_style_features)
        )

    return temp_style_loss


def style_transfer(
        base_img_path,
        style_img_path,
        output_img_path,
        content_weight=3e-2,
        style_weights=(20000, 500, 12, 1, 1),
        tv_weight=5e-2,
        content_layer='block4_conv2',
        style_layers=('block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1'),
        iterations=50
):

    width, height = tf.keras.preprocessing.image.load_img(base_img_path).size
    new_dims = (height, width)

    # Preprocess content and style images. Resizes the style image if needed.
    content_img = K.variable(read_image_as_tensor(base_img_path, new_dims))
    style_img = K.variable(read_image_as_tensor(style_img_path, new_dims))

    # Create an output placeholder with desired shape.
    # It will correspond to the generated image after minimizing the loss function.
    output_img = K.placeholder((1, height, width, 3))

    # Combine the 3 images into a single Keras tensor, for ease of manipulation
    # The first dimension of a tensor identifies the example/input.
    input_img = K.concatenate([content_img, style_img, output_img], axis=0)

    model = tf.keras.applications.vgg19.VGG19(input_tensor=input_img, weights='imagenet', include_top=False)

    # Get the symbolic outputs of each "key" layer (they have unique names).
    # The dictionary outputs an evaluation when the model is fed an input.
    layer_to_output_mapping = {layer.name: layer.output for layer in model.layers}

    # Extract features from the content layer
    content_features = layer_to_output_mapping[content_layer]

    # Extract the activations of the base image and the output image
    base_image_features = content_features[0, :, :, :]  # 0 corresponds to base
    combination_features = content_features[2, :, :, :]  # 2 coresponds to output

    # Calculate the feature reconstruction loss
    content_loss = content_weight * feature_reconstruction_loss(base_image_features, combination_features)

    # For each style layer compute style loss
    # The total style loss is the weighted sum of those losses
    # Extract as function
    style_loss = style_loss_for_all_layers(style_layers, style_weights, layer_to_output_mapping)
    # Compute total variational loss.
    tv_loss = tv_weight * total_variation_loss(output_img)
    # Composite loss
    total_loss = content_loss + style_loss + tv_loss

    # Compute gradients of output img with respect to total_loss
    grads = K.gradients(total_loss, output_img)
    outputs = [total_loss] + grads
    loss_and_grads = K.function([output_img], outputs)
    # Initialize the generated image from random noise
    x = np.random.uniform(0, 255, (1, height, width, 3)) - 128.

    # Loss function that takes a vectorized input image, for the solver
    def loss(x):
        x = x.reshape((1, height, width, 3))  # reshape
        return loss_and_grads([x])[0]

    # Gradient function that takes a vectorized input image, for the solver
    def grads(x):
        x = x.reshape((1, height, width, 3))  # reshape
        return loss_and_grads([x])[1].flatten().astype('float64')
    # Fit over the total iterations

    for i in progress_bar(range(iterations)):
        x, min_val, info = fmin_l_bfgs_b(loss, x.flatten(), fprime=grads, maxfun=20)
        # save current generated image
        if i % 10 == 0:
            img = tensor_to_image(x.copy(), width, height)
            fname = output_img_path + '_at_iteration_%d.png' % (i)

    return img
