""" interactive.py

    This file allows you to interactively give input to the generator and see the image generated

    Akshay Chitale, Siyuan Yu, and Haoran Lou
    5th May 2019
"""

from PIL import Image
import keras
import os
import numpy as np

import data_format as df
import data_prepare as dp


def interactive(save_dir='results'):
    generator = keras.models.load_model(os.path.join(save_dir, 'generator.hdf5'))
    _, _, _, vocab = dp.prepare()
    try:
        while True:
            print('Type some text, or press Ctrl+C to exit: ', end='')
            inp = input()
            inp = np.array([df.format_text(inp, vocab)])
            ex = generator.predict(inp).reshape((128, 128, 3))
            to_show = df.arr_to_image(ex)
            Image.fromarray(to_show).show()
    except KeyboardInterrupt:
        print('Goodbye.')

if __name__ == '__main__':
    # Interactively gives input to the trained model
    interactive()
