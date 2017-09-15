# this requires further testing.
import tensorflow as tf
from tensorflow import layers as l 
from tensorflow import nn
from tensorflow import image

def upsample_2x(x):
    shp = tf.shape(x)
    return image.resize_bilinear(x, (shp[1]*2, shp[2]*2))

def sep_bn(x, filts, strides=1):
    x = l.separable_conv2d(x, filts, (3, 3),strides=strides,padding='same', activation=nn.relu)
    return l.batch_normalization(x,fused=True)

def encode_block(x, filts, n_blocks=2):
    with tf.name_scope('strided_seperable'):
        x = sep_bn(x, filts, 2) 
        
    for i in range(n_blocks-1):
        with tf.name_scope('seperable'+str(i+i)):
            x = sep_bn(x, filts)
    return x

def decode_block(x_small, x_large, filts, n_blocks=2):
    with tf.name_scope('upsample'):
        x_up = upsample_2x(x_small)
        x = tf.concat((x_up, x_large), axis=-1)
        for i in range(n_blocks):
            with tf.name_scope('seperable'+str(i)):
                x = sep_bn(x, filts)
    return x

def make_encode_decode_net(inputs, num_classes, n_blocks, base=32, filter_list=None):
    if filter_list is None:
        filter_list = list()
        for i in range(n_blocks):
            filter_list.append(base*(2**i))

    with tf.name_scope('segmentation_net'):
        x0 = l.conv2d(inputs, base, (3,3), activation=nn.relu, padding='same')
        x0 = l.batch_normalization(x0)
        encode_layers = list()
        for i in range(n_blocks):
            name = 'encode_block' + str(i)
            with tf.name_scope(name):
                if i == 0:
                    encode_layers.append(encode_block(x0, filter_list[i]))
                else:
                    encode_layers.append(encode_block(encode_layers[i-1],filter_list[i]))

        encode_layers = [x0] + encode_layers
        encode_layers = list(reversed(encode_layers))
        filter_list = list(reversed(filter_list))

        for e, layer in enumerate(encode_layers[1:]):
            name = 'decode_block' + str(e)
            with tf.name_scope(name):
                if e == 0:
                    x = decode_block(encode_layers[0], layer,filter_list[e])
                else:
                    x = decode_block(x, layer, filter_list[e])

        x = l.conv2d(x, num_classes, (3,3), padding='same')  
    return x
