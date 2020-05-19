import numpy as np
from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.activations import sigmoid, relu
from keras.layers import Input, Reshape, Dense, LeakyReLU, UpSampling2D
from keras.layers.core import Flatten, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D

class Generator:

    @staticmethod
    def Default(original_output_shape, noise_depth, upsample_layers, kernel_size):
        # Model credits to Tiago Freitas: https://github.com/tensorfreitas/DCGAN-for-Bird-Generation/blob/master/traindcgan.py

        # Constant momentum for batch normalization
        momentum = 0.8

        model = Sequential()

        # Filter sizes for layers in which upsampling occurs
        ul_len = len(upsample_layers)

        # Reshape output to a multiple of 2^(number of upsample layers)
        output_shape = list(original_output_shape)
        if output_shape[0] % 2**ul_len != 0 or output_shape[1] % 2**ul_len != 0:
            output_shape[0] = int(np.ceil(output_shape[0] / 2**ul_len)) * 2**ul_len
            output_shape[1] = int(np.ceil(output_shape[1] / 2 ** ul_len)) * 2**ul_len
        output_shape = tuple(output_shape)

        # Get desired output shape
        output_size = np.prod(output_shape)
        num_channels = output_shape[2] if len(output_shape) > 2 else 1

        # Derive input shape
        dim_1, dim_2 = int(output_shape[0] / 2**ul_len), int(output_shape[1] / 2**ul_len)
        input_shape = (dim_1, dim_2, int(output_size / dim_1 / dim_2))

        # Add dense input layer
        # Creds to Mitchell Jolly for input layer https://github.com/mitchelljy/DCGAN-Keras/blob/master/DCGAN.py
        model.add(Dense(np.prod(input_shape), activation="relu", input_shape=(noise_depth,)))
        model.add(Reshape(input_shape))
        model.add(BatchNormalization(momentum=momentum))

        # Several upsampling convolution layers
        for layer in upsample_layers:
            model.add(UpSampling2D())
            model.add(Conv2D(filters=layer, kernel_size=kernel_size, padding='same'))
            model.add(Activation(relu))
            model.add(BatchNormalization(momentum=momentum))

        # Convolute the output of the upsampling trend
        model.add(Conv2D(num_channels, kernel_size=kernel_size, padding="same"))

        # Activate the last layer with the sigmoid function
        model.add(Activation(sigmoid))

        noise = Input(shape=(noise_depth,))
        model = Model(noise, model(noise))

        return model, output_shape


class Discriminator:

    @staticmethod
    def Default(input_shape, conv_layers, kernel_size):
        # Model credits to Tiago Freitas: https://github.com/tensorfreitas/DCGAN-for-Bird-Generation/blob/master/traindcgan.py

        # Constant momentum for batch normalization
        momentum = 0.5

        # Constant RELU activation leak
        relu_leak = 0.2

        model = Sequential()

        # Layer filter sizes
        for layer in conv_layers:
            model.add(Conv2D(filters=layer, kernel_size=kernel_size, padding='same', input_shape=input_shape))
            model.add(LeakyReLU(relu_leak))
            model.add(BatchNormalization(momentum=momentum))

        # Flatten the inputs and densify them
        model.add(Flatten())

        # Output layer
        model.add(Dense(1))
        model.add(Activation(sigmoid))

        image = Input(shape=input_shape)
        model = Model(image, model(image))

        return model


def GAN(noise_depth, generator, discriminator):

    # Input to generator
    noise = Input(shape=(noise_depth,))
    image = generator(noise)

    # Discriminate image
    discriminator.trainable = False
    valid = discriminator(image)

    # Form model
    model = Model(noise, valid)

    return model


def compile_model(model, lr, beta_1=0.5):
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=lr, beta_1=beta_1), metrics=None)
    return model