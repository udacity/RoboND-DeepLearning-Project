from tensorflow.contrib.keras.python import keras
from tensorflow.contrib.keras.python.keras import layers as l

from utils.separable_conv2d import SeparableConv2DKeras


# TODO implement a encode block, it should reduce spatial edge width by half
def encode_block(x, filts, n_blocks=2):
    pass

# TODO implement a decode block, it should increase spatial edge_width by half
def decode_block(x_small, x_large, filts, n_blocks=2):
    pass

def make_encode_decode_net(inputs, num_classes, n_blocks, base=32, filter_list=None):
    """Flexible semantic segmentation architecture using seperable convolutions"""

    if filter_list is None:
        filter_list = list()
        for i in range(n_blocks):
            filter_list.append(base * (2 ** i))

    x0 = l.Conv2D(base, 3, activation='relu', padding='same')(inputs)
    x0 = l.BatchNormalization()(x0)

    encode_layers = list()
    for i in range(n_blocks):
        if i == 0:
            encode_layers.append(encode_block(x0, filter_list[i]))
        else:
            encode_layers.append(encode_block(encode_layers[i - 1], filter_list[i]))

    encode_layers = [x0] + encode_layers
    encode_layers = list(reversed(encode_layers))
    filter_list = list(reversed(filter_list))

    for e, layer in enumerate(encode_layers[1:]):
        if e == 0:
            x = decode_block(encode_layers[0], layer, filter_list[e])
        else:
            x = decode_block(x, layer, filter_list[e])

    return l.Conv2D(num_classes, 3, activation='sigmoid', padding='same')(x)


def make_example_model():
    im_shape = (256, 256, 3)
    inputs = keras.layers.Input(im_shape)
    out_layer = make_encode_decode_net(inputs, num_classes=3, n_blocks=3, base=16)

    model = keras.models.Model(inputs=inputs, outputs=out_layer)
    model.compile(optimizer=keras.optimizers.Adam(0.001), loss='binary_crossentropy')
    return model
