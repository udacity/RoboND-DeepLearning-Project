import os
import json
from tensorflow.contrib.keras.python import keras 
from scipy import misc
from . import data_iterator
import numpy as np
import glob

def make_dir_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_network(your_model, your_weight_filename):
    config_name = 'config' + '_' + your_weight_filename
    weight_path = os.path.join('..', 'data', 'weights', your_weight_filename)
    config_path = os.path.join('..', 'data', 'weights', config_name)
    your_model_json = your_model.to_json()
    
    with open(config_path, 'w') as file:
        json.dump(your_model_json, file)  
        
    your_model.save_weights(weight_path) 
        
        
def load_network(your_weight_filename):
    config_name = 'config' + '_' + your_weight_filename
    weight_path = os.path.join('..', 'data', 'weights', your_weight_filename)
    config_path = os.path.join('..', 'data', 'weights', config_name)
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as file:
            json_string = json.load(file)  
            
        your_model = keras.models.model_from_json(json_string)
        
    else:
        raise ValueError('No config_yourmodel file found at {}'.format(config_path))
        
    if os.path.exists(weight_path):
        your_model.load_weights(weight_path)
        return your_model
    else:
        raise ValueError('No weight file found at {}'.format(weight_path))


def write_predictions_grade_set(model, out_folder_suffix,subset_name, grading_dir_name):
    validation_path = os.path.join('..', 'data', grading_dir_name, subset_name)
    file_names = sorted(glob.glob(os.path.join(validation_path, 'images', '*.jpeg')))

    output_path = os.path.join('..', 'data', 'runs', subset_name + '_' + out_folder_suffix)
    make_dir_if_not_exist(output_path)
    image_shape = model.layers[0].output_shape[1]

    for name in file_names:
        image = misc.imread(name)
        if image.shape[0] != image_shape:
             image = misc.imresize(image, (image_shape, image_shape, 3))
        image = data_iterator.preprocess_input(image.astype(np.float32))
        pred = model.predict_on_batch(np.expand_dims(image, 0))
        base_name = os.path.basename(name).split('.')[0]
        base_name = base_name + '_prediction.png'
        misc.imsave(os.path.join(output_path, base_name), np.squeeze((pred * 255).astype(np.uint8)))
    return validation_path, output_path
