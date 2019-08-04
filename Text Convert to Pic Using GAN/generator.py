""" generator.py

    This file contains the generator network for the GAN

    Akshay Chitale, Siyuan Yu, and Haoran Lou
    20th April 2019
"""

import keras

import consts

def build_generator(lr=0.001):
    """
    Builds the generator model

    :return: The compiled generator model
    """
    generator = keras.Sequential()

    # First layer is an embedding layer to convert from text indices in the vocabulary to numbers that can be used in a CNN
    generator.add(keras.layers.Embedding(consts.VOCAB_LEN, 4, input_length=consts.MAX_SENTENCE_LEN, embeddings_initializer='normal'))
    generator.add(keras.layers.Reshape((consts.MAX_SENTENCE_LEN, 4, 1)))

    # Convolution layers to get features
    generator.add(keras.layers.Conv2D(16, (2, 4), activation='elu'))
    generator.add(keras.layers.MaxPooling2D((3, 1)))
    generator.add(keras.layers.BatchNormalization())

    generator.add(keras.layers.Conv2D(64, (8, 1), activation='elu'))
    generator.add(keras.layers.MaxPooling2D((10, 1)))
    generator.add(keras.layers.BatchNormalization())

    # Flatten and dense layers
    generator.add(keras.layers.Flatten())
    generator.add(keras.layers.Dropout(0.5))
    generator.add(keras.layers.Dense(900, activation='elu'))
    generator.add(keras.layers.Reshape((30, 30, 1, 1)))

    # Transposed convolutions to bring it back up to an image
    generator.add(keras.layers.UpSampling3D((2, 2, 1)))
    generator.add(keras.layers.Conv3DTranspose(16, (4, 4, 2)))
    generator.add(keras.layers.BatchNormalization())

    generator.add(keras.layers.UpSampling3D((2, 2, 1)))
    generator.add(keras.layers.Conv3DTranspose(1, (3, 3, 2)))
    generator.add(keras.layers.BatchNormalization())


    # Make output the right shape
    #assert generator.layers[-1].output_shape == (None, *consts.IMAGE_INPUT_SHAPE)

    # Compile and return
    generator.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adam(lr=lr), metrics=['accuracy'])

    return generator

if __name__ == '__main__':
    # Test script to see the generator model summary
    model = build_generator()
    model.summary()
