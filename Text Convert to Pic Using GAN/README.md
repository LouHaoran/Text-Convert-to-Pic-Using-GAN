#Semester Project: Image Generation based on Description
**Akshay Chitale, Siyuan Yu, and Haoran Lou**

**CS 6375.001 Spring 2019, Dr. Anjum Chida**

**3rd May 2019**

## Dataset Notes:
To download the dataset:
* Make a directory called "coco"
* Go to [http://cocodataset.org/#download](http://cocodataset.org/#download)
* Download "2017 Val images [5K/1GB]"
    * Unzip the file into a folder in the "coco" directory called "val2017"
* Download "2017 Test images [41K/6GB]"
    * Unzip the file into a folder in the "coco" directory called "test2017"
* Download "2017 Train images [118K/18GB]"
    * Unzip the file into a folder in the "coco" directory called "train2017"
* Download "2017 Train/Val annotations [241MB]"
    * Unzip the file into a folder in the "coco" directory called "annotations_trainval2017"
* Download "2017 Testing Image info [1MB]"
    * Unzip the file into a folder in the "coco" directory called "image_info_test2017"
    
Some stats about the training set:
* There are 37226 captions (so not all captions have images, need to filter out)
* The maximum sentence length is 250
* The vocabulary is in `vocab.txt`. The size of the vocabulary is 37226

## Requirements
This code must be run using Python3. In addition, it uses the following Python modules:

* PIL
* imageio
* keras
* numpy
* os
* string
* tqdm

Any missing package can be installed using pip. Note that PIL is installed using `pip3 install pillow`

## Training:
To run training, first delete the previous `results` folder if it exists. Then, execute `main.py`:

`$ python3 main.py`

To execute with different parameters, such as a different number of epochs, edit `main.py` and execute. See the docstring of `train_gan` in gan.py for more information

## Interactive execution:
To run an interactive session on a pre-trained network, execute `interactive.py`:

`$ python3 interactive.py`

## Files:
* `coco` - A directory for the COCO dataset
* `consts.py` - Some constants used throughout the project
* `data` - Generated files from `data_prepare.py` of the dataset
    * `train.txt` - Pairs of image path, caption for the training set
    * `val.txt` - Pairs of image path, caption for the validation set
    * `vocab.txt` - A list of words in the vocabulary
* `data_format.py` - Formats data for input and output to and from the network
* `data_prepare.py` - Generates the test, validation, and training set, as well as the vocab list
* `discriminator.py` - Defines the discriminator
* `gan.py` - Defines the GAN and the training process
* `generator.py` - Defines the generator
* `interactive.py` - Lets you interactively give input to the trained generator
* `main.py` - Calls `train_gan` with some small parameters
* `ProjectReport.pdf` - The report for this project
* `README.md` - The markdown file of this README
* `results` - The results generated from the most recent run. Delete this folder before running again so that the results do not mix
    * `examples` - A directory of some validation examples saved during training at each epoch
    * `discriminator_loss.csv` - The loss over epoch for the discriminator
    * `discriminator.hdf5` - The discriminator model
    * `gan_loss` - The loss over epoch for the GAN (therefore, the generator)
    * `generator.hdf5` - The generator model
