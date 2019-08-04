""" data_format.py

    This file contains functions to process input and output data

    Akshay Chitale, Siyuan Yu, and Haoran Lou
    27th April 2019
"""

import numpy as np
from PIL import Image
import string

import consts


def format_text(text, v):
    """
    Formats text as an array with a max len of consts.MAX_SENTENCE_LEN with the index being the index in the vocab

    :param text: A string of text
    :param v: A dictionary of {word: index} for the vocab
    :return: A numpy array representing this string of text
    """

    # Get rid of punctuation and etc, also limit length to consts.MAX_SENTENCE_LEN
    txt = [s.translate(str.maketrans('', '', string.punctuation)) for s in text.split()[:consts.MAX_SENTENCE_LEN]]

    try:
        ret = [v[word] for word in txt]
    except KeyError:
        # Try one by one if failed
        ret = [][:]
        for word in txt:
            try:
                ret.append(v[word])
            except KeyError:
                ret.append(0)

    return np.pad(np.array(ret, dtype=float), pad_width=(0, consts.MAX_SENTENCE_LEN-len(ret)), mode='constant')


def read_image(_file):
    """
    Reads an image as a 128 x 128 numpy array

    :param _file: The file to read the image from
    :return: A numpy array of shape (128, 128, 3)
    """
    im = Image.open(_file)
    im = im.resize((128, 128), Image.ANTIALIAS)

    # Ensure 3 channels
    if im.mode != 'RGB':
        im = im.convert('RGB')
    
    return np.array(im, dtype=float)/255 # Let max value be 1


def arr_to_image(im):
    """
    Converts the network's image arrays into an image array that can be saved

    :param im: The network image array, floats
    :return: Showable images, uint8
    """
    capped = np.maximum(0, np.minimum(1, im))
    return np.array(capped*255, dtype=np.uint8)


if __name__ == "__main__":
    # Small test script
    import data_prepare as dp
    train, val, test, vocab = dp.prepare()

    text = format_text(train[0][1], vocab)
    print("TXT SHAPE: ", end='')
    print(text.shape)
    print(text)

    img = read_image(train[0][0])
    print("IMG SHAPE: ", end='')
    print(img.shape)
    Image.fromarray(arr_to_image(img)).show()
    