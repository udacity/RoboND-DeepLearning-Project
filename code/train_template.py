import argparse
import os

from tensorflow.contrib.keras.python import keras

import make_model
from utils import data_iterator

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('save_model_file',
                        help='the file to save the model to')

    parser.add_argument('pretrain_model_file',
                        help='if you dont have one just enter anything that is not a model file')

    args = parser.parse_args()
    model_save_file = args.save_model_file
    model_load_file = args.pretrain_model_file

    if args.pretrain_model_file is not None:
        model_path = os.path.join('..', 'data', 'weights', model_load_file)

    # TODO instantiate data iterators and implement a training loop
