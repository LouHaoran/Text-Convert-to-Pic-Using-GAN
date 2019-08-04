""" discriminator.py

    This file contains the discriminator network for the GAN

    Akshay Chitale, Siyuan Yu, and Haoran Lou
    20th April 2019
"""

import keras

import consts

def build_discriminator(lr=0.001):
    """
    Builds the discriminator

    :param lr: The learning rate
    :return:
    """
    discriminator = keras.Sequential()
    discriminator.add(keras.layers.Reshape(consts.IMAGE_INPUT_SHAPE, input_shape=consts.IMAGE_INPUT_SHAPE))

    # Convolutional layers to get features
    discriminator.add(keras.layers.Conv3D(32, (3, 3, 2), padding='same', activation='relu'))
    discriminator.add(keras.layers.MaxPooling3D((4, 4, 1)))
    discriminator.add(keras.layers.BatchNormalization())

    discriminator.add(keras.layers.Conv3D(2, (3, 3, 2), padding='same', activation='relu'))
    discriminator.add(keras.layers.MaxPooling3D((4, 4, 1)))
    discriminator.add(keras.layers.BatchNormalization())

    # Flatten and Dense layers
    discriminator.add(keras.layers.Flatten())
    discriminator.add(keras.layers.Dense(128, activation='relu'))
    discriminator.add(keras.layers.Dropout(0.5))

    # Softmax on last layer
    discriminator.add(keras.layers.Dense(2, activation='softmax'))

    discriminator.compile(optimizer=keras.optimizers.Adam(lr=lr), loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

    return discriminator

if __name__ == '__main__':
    # Test script to see the discriminator model summary
    model = build_discriminator()
    model.summary()