from tensorflow.contrib.keras.python import keras
from tensorflow.contrib.keras.python.keras import layers as l
from utils.separable_conv2d import SeparableConv2DKeras
from functools import partial
import json
import os

import network_config
from utils import preprocess_ims

def conv2d_layer(x, filters,kernel_size=3, strides=1, batch_norm=True):
    x = l.Conv2D(filters, kernel_size, strides=strides, padding='same')(x) 
    if batch_norm:
        x = l.BatchNormalization()(x)
    return x


def separable2d_layer(x, filters,kernel_size=3, strides=1, batch_norm=True):
    x = SeparableConv2DKeras(filters, kernel_size, strides=strides,padding='same')(x) 
    if batch_norm:
        x = l.BatchNormalization()(x)
    return x


def downsample(x, layer_fn, downsample_type):
    if downsample_type == 'strided_convolution':
        x = layer_fn(x, strides=2)
    else:
        x = layer_fn(x)
        if downsample_type == 'max_pooling':
            x = l.MaxPooling2D()(x)
        elif downsample_type == 'average_pooling':
            x = l.AveragePooling2D()(x)
    return x


def encode_block(x, layer_fn,downsample_type, n_layers):
    x = downsample(x, layer_fn, downsample_type)
    for i in range(n_layers-1):
        x = layer_fn(x)
    return x


def make_filter_list(base, n_blocks_per_side):
    filter_list = list()
    for i in range(n_blocks_per_side):
        filter_list.append(base * (2 ** i))
    return filter_list + list(reversed(filter_list))


def make_constant_list(value, n_blocks_per_side):
    return [value] * 2 * n_blocks_per_side


def process_maybe_lists(config_dict):
    n_blocks_per_side = config_dict['n_blocks_per_side']
    if not isinstance(config_dict['batch_norm'], list):
        batch_norms = make_constant_list(config_dict['batch_norm'], n_blocks_per_side)
    else:
        batch_norms = config_dict['batch_norm']
    
    if not isinstance(config_dict['block_size'], list):
        block_sizes = make_constant_list(config_dict['block_size'], n_blocks_per_side)
    else:
        block_sizes = config_dict['block_size']
    
    if not isinstance(config_dict['n_filters'], list):
        n_filters = make_filter_list(config_dict['n_filters'], n_blocks_per_side)
    else:
        n_filters = config_dict['n_filters']

    return list(zip(n_filters, block_sizes, batch_norms))


def make_partial_layer(conv_type, filters, batch_norm):
    if conv_type == 'regular':
        layer_fn = partial(conv2d_layer, filters=filters,kernel_size=3, batch_norm=batch_norm)
    else:
        layer_fn = partial(separable2d_layer, filters=filters, kernel_size=3, batch_norm=batch_norm)
    return layer_fn


def upsample(x, upsample_type, n_out_filts=None):
    if upsample_type == 'nearest_neighbor':
        x = l.UpSampling2D(size=(2, 2))(x)
    else:
        # TODO the output size here needs to be computed
        x = l.Conv2DTranspose(n_out_filts, 3, strides=2, padding='same') 
    return x


def decode_block(x, skip_layer, layer_fn, upsample_type, n_layers, n_out_filts=None):
    x = upsample(x, upsample_type, n_out_filts)
    if upsample_type == 'nearest_neighbor':
        x = layer_fn(x)
    
    x = l.concatenate([x, skip_layer])
    for i in range(n_layers-1):
        x = layer_fn(x)
    return x


def build_segmentation_network(config_dict):
    lists = process_maybe_lists(config_dict) 

    n_blocks = config_dict['n_blocks_per_side']
    im_edge =  config_dict['image_resolution']
    inputs = l.Input((im_edge, im_edge, 3))

    encode_blocks = list()
    x = conv2d_layer(inputs, lists[0][0], True)

    encode_blocks.append(x)
    for e, tup in enumerate(lists):
        if e < n_blocks: 
            layer_fn = make_partial_layer(config_dict['convolution_type'], tup[0], tup[2])
            x = encode_block(x, layer_fn, config_dict['downsample_method'], tup[1])
            encode_blocks.append(x)
        else:
            layer_fn = make_partial_layer(config_dict['convolution_type'], tup[0], tup[2])
            skip_layer = encode_blocks[n_blocks - e - 2]
            x = decode_block(x, skip_layer, layer_fn, config_dict['upsample_method'], tup[1])

    x = l.Conv2D(config_dict['n_classes'], 1, activation=config_dict['last_layer_activation'], padding='same')(x)
    return x, inputs 



def config_to_json(config_dict, file_name):
    file_name = os.path.join('..', 'data', 'model_info', file_name +'.json')
    with open(file_name, 'w') as f:
        json.dump(config_dict, f)


def network_from_json(file_name):
    with open(file_name) as f:
        config_dict = json.load(f)

    x = build_segmentation_network(config_dict)
    return x


def make_example_network():    
    config_dict = network_config.make_config()
    config_dict = check_config(config_dict)

    x, inputs = build_segmentation_network(config_dict)
    model = keras.models.Model(inputs=inputs, outputs=x)

    loss_strings = {'softmax':'categorical_crossentropy', 'sigmoid':'binary_crossentropy'} 

    model.compile(optimizer=keras.optimizers.Adam(config_dict['learning_rate']), 
            loss=loss_strings[config_dict['last_layer_activation']])

    plot_keras_model(model, config_dict['model_name'])
    config_to_json(config_dict,config_dict['model_name'])
    return model

def plot_keras_model(model, name):
    base_path = os.path.join('..', 'data', 'model_info')
    preprocess_ims.make_dir_if_not_exist(base_path)
    keras.utils.vis_utils.plot_model(model, os.path.join(base_path, name))
    keras.utils.vis_utils.plot_model(model, os.path.join(base_path, name +'_with_shapes'), show_shapes=True)


def is_list_or_int(obj, name):
    if not (isinstance(obj, list) or isinstance(obj, int)):
        raise ValueError('{}, must either be a list or an integer it is a {}'.format(name, type(obj)))


def check_list_len(obj, n_blocks,  name):
    if isinstance(obj, list):
        if len(obj) != 2*n_blocks:  
            raise ValueError('If passing {} as a list the length must be equal to 2*n_blocks_per_side({}), but it has length {}'.format(name, 2*n_blocks, len(obj)))

def check_list(config_dict, name):
    is_list_or_int(config_dict[name], name)
    check_list_len(config_dict[name], config_dict['n_blocks_per_side'], name)


def check_string(config_dict, name, allowed_values):
    if config_dict[name] not in allowed_values:
        raise ValueError('last_layer_activation must either be one of {} but it is {}'.format(str(allowed_values), config_dict[name]))


def check_config(config_dict):
    check_list(config_dict, 'n_filters')
    check_list(config_dict, 'batch_norm')
    check_list(config_dict, 'block_size')
    
    check_string(config_dict, 'downsample_method', ['strided_convolution', 'max_pooling', 'average_pooling'])
    check_string(config_dict, 'upsample_method', ['nearest_neighbor', 'transposed_convolution'])
    check_string(config_dict, 'convolution_type', ['separable', 'regular'])
    check_string(config_dict, 'last_layer_activation', ['sigmoid', 'softmax'])

    allowed_resolutions = [256, 224, 192, 160, 128, 96, 64, 32]
    check_string(config_dict, 'image_resolution', allowed_resolutions) 
    
    if config_dict['block_size'] < 1:
        raise ValueError('block_size must at least be 1 it is is {}'.format(config_dict['block_size'])) 

    if config_dict['image_resolution'] // 2 ** config_dict['n_blocks_per_side'] < 4:
        raise ValueError('The smallest allowed spatial dimensions for the inner convolution layer is 4, either increase the image resolution or decrease n_blocks_per_side')

    return config_dict
