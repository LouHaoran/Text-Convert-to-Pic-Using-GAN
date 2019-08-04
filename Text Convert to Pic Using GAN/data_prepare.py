""" data_prepare.py

    This file contains functions to prepare the dataset for training/testing

    Akshay Chitale, Siyuan Yu, and Haoran Lou
    20th April 2019
"""

import os
import ijson
import string

import consts

def safe_mkdir(pth):
    """
    Makes the directory if it does not exist

    :param pth: The path to the directory
    :return: None
    """
    try:
        os.mkdir(pth)
    except FileExistsError:
        pass


def _load_set(annotation_f, img_dir, fname, force=False):
    """
    Loads one set of captions

    :param annotation_f: The JSON file of annotations in COCO format
    :param img_dir: The directory in which images are held
    :param fname: The file to which to save the data read
    :param force: Whether to force regeneration
    :return: The set as (image path, caption) tuples
    """
    try:
        if force:
            raise FileNotFoundError()
        with open(fname, 'r') as f:
            # Only split on first comma in case of commas in the text
            ret = [x.strip().split(',', 1) for x in f.readlines() if x]
    except FileNotFoundError:
        # Need to generate
        print('File %s not found. Generating file...' % fname)
        ret = [][:]
        count = 0
        with open(annotation_f, 'r') as f:
            data = ijson.items(f, 'annotations.item')
            for datum in data:
                img_pth = os.path.join(img_dir, '%012d.jpg' % datum['image_id'])
                if os.path.isfile(img_pth):
                    # Print periodically when image exists
                    count += 1
                    if count % 10000 == 0:
                        print('Processed %d images' % count)
                    img_caption = datum['caption'].strip().replace('\n', '')
                    ret.append((img_pth, img_caption))
        print('Saving %s' % fname)
        with open(fname, 'w') as f:
            for x in ret:
                f.write('%s,%s\n' % x)
    return ret


def prepare(pth='data', force=False, smaller=1.0):
    """
    Prepares the needed information for training

    Each returned set is a list of (image path, caption) tuples

    The vocab set is a list of the vocab words

    :param pth: The path to the data
    :param force: Whether to force regeneration
    :param smaller: Fraction by which to make the data sets smaller
    :return: (training set, validation set, test set, vocab map)
    """
    assert smaller > 0.0
    assert smaller <= 1.0

    # Ensure that dir exists
    print('Preparing data...')
    safe_mkdir(pth)

    # Load validation data
    print('Loading validation set...')
    val = _load_set(annotation_f=os.path.join(consts.COCO_DIR, consts.COCO_VAL_ANNOTATIONS),
                    img_dir=os.path.join(consts.COCO_DIR, consts.COCO_VAL_IMG_DIR),
                    fname=os.path.join(pth, 'val.txt'), force=force)

    # Load training data
    print('Loading training set...')
    train = _load_set(annotation_f=os.path.join(consts.COCO_DIR, consts.COCO_TRAIN_ANNOTATIONS),
                      img_dir=os.path.join(consts.COCO_DIR, consts.COCO_TRAIN_IMG_DIR),
                      fname=os.path.join(pth, 'train.txt'), force=force)

    # Load test data
    # FIXME No annotations exist for test data, so using validation instead
    print('Loading test set...')
    test = val[:]

    # Load vocabulary
    print('Loading vocabulary...')
    try:
        if force:
            raise FileNotFoundError()
        with open(os.path.join(pth, 'vocab.txt'), 'r') as f:
            vocab = [ln.strip() for ln in f.readlines() if ln]
    except FileNotFoundError:
        # Generate vocab list
        print('File %s not found. Generating file...' % os.path.join(pth, 'vocab.txt'))
        vocab = set()
        for _, caption in train:
            # Avoid duplicates
            vocab.update(s.translate(str.maketrans('', '', string.punctuation)) for s in caption.split())
        # Maintain order
        vocab = [s for s in vocab]
        print('Saving %s'  % os.path.join(pth, 'vocab.txt'))
        with open(os.path.join(pth, 'vocab.txt'), 'w') as f:
            for v in vocab:
                f.write('%s\n' % (v))

    # Determine longest caption and vocab size and make sure they fit in the network
    print('Checking vocabulary...')
    longest = max(train, key=(lambda c: len(c[1].split())))
    print('Longest caption: %d' % len(longest[1]))
    assert len(longest[1]) <= consts.MAX_SENTENCE_LEN
    vlen = len(vocab)
    print('Vocab length: %d' % vlen)
    assert vlen <= consts.VOCAB_LEN

    # Return vocab as a map for faster lookup
    print('Making vocab into a dictionary')
    vocab_map = {word: index for index, word in enumerate(vocab)}

    # Trim data sets
    if smaller < 1.0:
        print('Trimming datasets to %.3f of their size' % smaller)
        val = val[:int(len(val) * smaller)]
        train = train[:int(len(train) * smaller)]
        test = test[:int(len(test) * smaller)]

    print('Done preparing data.')

    return train, val, test, vocab_map


if __name__ == "__main__":
    # Small test script
    train, val, test, vocab = prepare()
    print('TRAIN LEN: %d' % len(train))
    print('VAL LEN: %d' % len(val))
    print('TEST LEN: %d' % len(test))
