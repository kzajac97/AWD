# AWD

This repository contains implementation of various generative models as a part of AWD project. <br>

Implemented models are:
* Neural Style Matching Model
* Adversarial Generative Network
* Neural Style Transfer

### Neural Style Matching Model

Generative model implementing old fashioned type of learning, where generator is constructed taking in random vector from latent distribution (usually normal) and the loss function is computed simply as $L_2$ over the dataset. It can generate images which tend to be averaged out versions of the dataset in a given class.

### Adversarial Generative Network

Classical GAN model fitted using MNIST dataset.

### Neural Style Transfer

Algorithm using feature extraction mechanism based on convolutional neural network to change photographs into painting-like images. It is described in this [paper](https://arxiv.org/pdf/1508.06576.pdf) and [tensorflow tutorial](https://www.tensorflow.org/tutorials/generative/style_transfer). 
