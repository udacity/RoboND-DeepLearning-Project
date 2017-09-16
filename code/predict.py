import glob
import numpy as np
import os
import sys
from scipy import misc

import make_model
from utils import data_iterator


def make_dir_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise ValueError('predict.py requires a model file name, and output folder name cli input')

    model_file = sys.argv[1]
    output_folder = sys.argv[2]
    output_path = os.path.join('..', 'data', 'runs', output_folder)

    make_dir_if_not_exist(output_path)
    model = make_model.make_example_model()
    model.load_weights(os.path.join('..', 'data', 'weights', model_file))
    
    data_folder = os.path.join('..', 'data', 'validation')
    file_names = sorted(glob.glob(os.path.join(data_folder, 'images', '*.jpeg')))

    for name in file_names:
        image = misc.imread(name)
        image = data_iterator.preprocess_input(image.astype(np.float32))
        pred = model.predict_on_batch(np.expand_dims(image, 0))
        base_name = os.path.basename(name).split('.')[0]
        base_name = base_name + '_prediction.png'

        misc.imsave(os.path.join(output_path, base_name), np.squeeze((pred * 255).astype(np.uint8)))
