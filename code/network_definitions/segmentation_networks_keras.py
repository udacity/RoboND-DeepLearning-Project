import tensorflow as tf
from tensorflow.contrib.keras.python.keras import layers as l


# a configurable neural network for semantic segmentation
def encode_block(x, filts, n_blocks=2):
    """Reduces spatial resolution using strided convolution"""
    x = l.SeparableConv2D(filts, 3, strides=2, activation='relu', padding='same')(x)
    x = l.BatchNormalization()(x)
    for i in range(n_blocks-1):
        x = l.SeparableConv2D(filts, 3, activation='relu', padding='same')(x)
        x = l.BatchNormalization()(x)
    return x

def decode_block(x_small,x_large, filts, n_blocks=2):
    """Increases spatial resolution nearest neighbor upsampling"""
    up = l.UpSampling2D(size=(2,2))(x_small)
    x = l.concatenate([up, x_large], axis=-1)

    for i in range(n_blocks):
        x = l.SeparableConv2D(filts, 3, activation='relu', padding='same')(x)
        x = l.BatchNormalization()(x)

    return x

def make_encode_decode_net(inputs, num_classes, n_blocks, base=32, filter_list=None):
    """Flexible semantic segmentation architecture using seperable convolutions"""

    if filter_list is None:
        filter_list = list()
        for i in range(n_blocks):
            filter_list.append(base*(2**i))

    x0 = l.Conv2D(base, 3, activation='relu', padding='same')(inputs)
    x0 = l.BatchNormalization()(x0)
    
    encode_layers = list()
    for i in range(n_blocks):
        if i == 0:
            encode_layers.append(encode_block(x0, filter_list[i]))
        else:
            encode_layers.append(encode_block(encode_layers[i-1],filter_list[i]))

    encode_layers = [x0] + encode_layers
    encode_layers = list(reversed(encode_layers))
    filter_list = list(reversed(filter_list))

    for e, layer in enumerate(encode_layers[1:]):
        if e == 0:
            x = decode_block(encode_layers[0], layer,filter_list[e])
        else:
            x = decode_block(x, layer, filter_list[e])

    return l.Conv2D(num_classes, 3, activation='sigmoid', padding='same')(x)


# The following are implementations of similar networks to above but in a form
# that shows the structure of the network more clearly


# an implementations of the same network using encode and decode blocks
def make_encode_decode_net_alt1(inputs, num_classes):
    x0 = l.Conv2D(base, 3, activation='relu', padding='same')(inputs)
    x0 = l.BatchNormalization()(x0)

    x1 = encode_block(x0, 32)
    x2 = encode_block(x1, 64)
    x3 = encode_block(x2, 128)

    x = decode_block(x3, x2, 128)
    x = decode_block(x, x1, 64)
    x = decode_block(x, x0, 32)
    
    # use sigmoid activation and binary crossentropy when on pixel can belong to multiple classes
    x = l.Conv2D(num_classes, 3, activation='sigmoid', padding='same')(x)

    # use softmax activation and categorical crossentropy loss when the classes are disjoint 
    # x = l.Conv2D(num_classes, 3, activation='softmax', padding='same')(x)
    return x


# A slightly expanded implementation
def make_encode_decode_net_alt2(inputs, num_classes):
    x0 = l.Conv2D(base, 3, activation='relu', padding='same')(inputs)
    x0 = l.BatchNormalization()(x0)

    x1 = l.SeparableConv2D(32, 3, strides=2, activation='relu', padding='same')(x0)
    x1 = l.BatchNormalization()(x1)
    x1 = l.SeparableConv2D(32, 3, activation='relu', padding='same')(x1)
    x1 = l.BatchNormalization()(x1)

    x2 = l.SeparableConv2D(64, 3, strides=2, activation='relu', padding='same')(x2)
    x2 = l.BatchNormalization()(x2)
    x2 = l.SeparableConv2D(64, 3, activation='relu', padding='same')(x2)
    x2 = l.BatchNormalization()(x2)

    x3 = l.SeparableConv2D(128, 3, strides=2, activation='relu', padding='same')(x2)
    x3 = l.BatchNormalization()(x3)
    x3 = l.SeparableConv2D(128, 3, activation='relu', padding='same')(x3)
    x3 = l.BatchNormalization()(x3)

    up = l.UpSampling2D(size=(2,2))(x3)
    mrg = l.concatenate([up, x2], axis=-1)
    x = l.SeparableConv2D(filts, 3, activation='relu', padding='same')(mrg)
    x = l.BatchNormalization()(x)
    x = l.SeparableConv2D(filts, 3, activation='relu', padding='same')(x)
    x = l.BatchNormalization()(x)

    up = l.UpSampling2D(size=(2,2))(x)
    mrg = l.concatenate([up, x1], axis=-1)
    x = l.SeparableConv2D(filts, 3, activation='relu', padding='same')(mrg)
    x = l.BatchNormalization()(x)
    x = l.SeparableConv2D(filts, 3, activation='relu', padding='same')(x)
    x = l.BatchNormalization()(x)

    up = l.UpSampling2D(size=(2,2))(x)
    mrg = l.concatenate([up, x0], axis=-1)
    x = l.SeparableConv2D(filts, 3, activation='relu', padding='same')(mrg)
    x = l.BatchNormalization()(x)
    x = l.SeparableConv2D(filts, 3, activation='relu', padding='same')(x)
    x = l.BatchNormalization()(x)

    # use sigmoid activation and binary crossentropy when on pixel can belong to multiple classes
    x = l.Conv2D(num_classes, 3, activation='sigmoid', padding='same')(x)

    # use softmax activation and categorical crossentropy loss when the classes are disjoint 
    # x = l.Conv2D(num_classes, 3, activation='softmax', padding='same')(x)
    return x
