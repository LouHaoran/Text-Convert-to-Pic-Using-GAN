""" consts.py

    This file contains configuration constants for the network

    Akshay Chitale, Siyuan Yu, and Haoran Lou
    23th April 2019
"""

MAX_SENTENCE_LEN = 250
VOCAB_LEN = 37226
IMAGE_INPUT_SHAPE = (128, 128, 3, 1)

COCO_DIR = 'coco'
COCO_TRAIN_ANNOTATIONS = 'annotations_trainval2017/captions_train2017.json'
COCO_VAL_ANNOTATIONS = 'annotations_trainval2017/captions_val2017.json'
COCO_TEST_ANNOTATIONS = 'image_info_test2017/image_info_test2017.json'
COCO_TRAIN_IMG_DIR = 'train2017'
COCO_VAL_IMG_DIR = 'val2017'
COCO_TEST_IMG_DIR = 'test2017'