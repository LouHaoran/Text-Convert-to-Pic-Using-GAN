""" gan.py

    This file combines the generator and the discriminator for the GAN

    Akshay Chitale, Siyuan Yu, and Haoran Lou
    23rd April 2019
"""

import keras
import imageio
import numpy as np
import os
from tqdm import tqdm

import consts
import generator as gen
import discriminator as dis
import data_prepare as dp
import data_format as df

def build_gan(generator, discriminator):
    """
    Builds the GAN model

    :param generator: The compiled generator model with input consts.TEXT_INPUT_SHAPE and output
    consts.IMAGE_INPUT_SHAPE
    :param discriminator: The compiled discriminator model with input consts.IMAGE_INPUT_SHAPE and one output node
    :return: The compiled GAN model
    """
    # Link generator and discriminator
    discriminator.trainable = False
    _input = keras.layers.Input((consts.MAX_SENTENCE_LEN,))
    intermediate = generator(_input)
    _output = discriminator(intermediate)

    # Make GAN model
    gan = keras.Model(inputs=_input, outputs=_output)
    gan.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

    return gan


def train_gan(epochs = 1, batch_size=512, smaller=1.0, save_dir='results', load=False, examples=1,
              imformat=df.read_image, txtformat=df.format_text, on_epoch=None):
    """
    Traing a GAN

    :param epochs: Number of epochs to do
    :param batch_size: The batch size when running
    :param smaller: The amount by which to make the dataset smaller
    :param save_dir: The directory to save the results
    :param load: Whether to load the model
    :param examples: The number of examples to save
    :param imformat: The function to format images
    :param txtformat: The function to format text
    :param on_epoch: A function to run on every epoch
    :return:
    """
    # For saved models
    dp.safe_mkdir(save_dir)
    dp.safe_mkdir(os.path.join(save_dir, 'examples'))

    # Load data
    train, val, _, vocab = dp.prepare(smaller=smaller)
    print('TRAIN SIZE %d' % (len(train),))
    print('VAL SIZE %d' % (len(val),))

    # Check values
    num_batch = int(np.ceil(len(train) / batch_size))
    num_val = int(np.ceil(len(val) / batch_size))
    assert len(train) >= batch_size

    # Prepare examples
    assert examples <= len(val)
    assert examples >= 0
    for i in range(examples):
        dp.safe_mkdir(os.path.join(save_dir, 'examples', '%d' % i))
        with open(os.path.join(save_dir, 'examples', '%d' % i, 'caption.txt'), 'w') as f:
            f.write(val[i][1])
        imageio.imwrite(os.path.join(save_dir, 'examples', '%d' % i, 'correct.jpg'), imageio.imread(val[i][0]))

    # Print that we're starting
    print('Training on %d images for %d epochs' % (len(train), epochs))
    print('Will save %d examples every epoch' % (examples,))

    # Create GAN
    generator = keras.models.load_model(os.path.join(save_dir, 'generator.hdf5')) if load else gen.build_generator()
    discriminator = keras.models.load_model(os.path.join(save_dir, 'discriminator.hdf5')) if load else dis.build_discriminator()
    gan = build_gan(generator, discriminator)

    # For every epoch, for every batch
    for e in range(1, epochs + 1):
        print("Epoch %d" % e)
        idx = 0
        for _ in tqdm(range(num_batch)):
            # Get the input for this batch
            l = idx * batch_size  # low bound of batch
            h = (idx + 1) * batch_size # high bound of batch
            idx += 1

            # Get text, np images
            current_train = train[l:h]
            batch_im = np.array(list(imformat(impth) for impth, _ in current_train))
            batch_cap = np.array(list(txtformat(caption, vocab) for _, caption in current_train))

            # Generate images to be compared
            gen_im = generator.predict(batch_cap)

            # Training data for discriminator
            batch_im = batch_im.reshape((len(current_train), *consts.IMAGE_INPUT_SHAPE))
            all_data = np.concatenate([batch_im, gen_im])
            del batch_im
            all_labels = np.zeros(2 * len(current_train))
            all_labels[:len(current_train)] = 1  # Almost one
            all_labels = keras.utils.to_categorical(all_labels, 2)

            # Train discriminator
            discriminator.trainable = True
            dloss, dacc = discriminator.train_on_batch(all_data, all_labels)
            del all_data
            del all_labels

            # Train generator now on the original data
            discriminator.trainable = False
            batch_labels = keras.utils.to_categorical(np.ones(len(current_train)), 2)  # Trick into thinking generated images are the real ones
            gloss, gacc = gan.train_on_batch(batch_cap, batch_labels)
            del batch_cap
            del batch_labels

        # Save models after each epoch
        generator.save(os.path.join(save_dir, 'generator.hdf5'))
        discriminator.save(os.path.join(save_dir, 'discriminator.hdf5'))

        # Save the losses from the end of the last epoch
        with open(os.path.join(save_dir, 'discriminator_loss.csv'), 'w' if e == 1 else 'a') as f:
            if e == 1:
                f.write('Epoch,Loss,Accuracy\n')
            f.write('%d,%f,%f\n' % (e, dloss, dacc))
        with open(os.path.join(save_dir, 'gan_loss.csv'), 'w' if e == 1 else 'a') as f:
            if e == 1:
                f.write('Epoch,Loss,Accuracy\n')
            f.write('%d,%f,%f\n' % (e, gloss, gacc))

        # Save an example
        for i in range(examples):
            inp = np.array([df.format_text(val[i][1], vocab)])
            ex = generator.predict(inp).reshape((128, 128, 3))
            to_save = df.arr_to_image(ex)
            imageio.imwrite(os.path.join(save_dir, 'examples', '%d' % i, '%d.jpg' % e), to_save)

        # Other ops
        if on_epoch is not None:
            on_epoch(generator=generator, discriminator=discriminator, train=train, val=val, test=None, vocab=vocab)

if __name__ == '__main__':
    # Test script to see the GAN model summary
    model = build_gan(gen.build_generator(), dis.build_discriminator())
    model.summary()