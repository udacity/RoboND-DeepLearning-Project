# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# modified by Devin Anzelmo 2017

from tensorflow.contrib.keras.python.keras import activations
from tensorflow.contrib.keras.python.keras import backend as K
from tensorflow.contrib.keras.python.keras import constraints
from tensorflow.contrib.keras.python.keras import initializers
from tensorflow.contrib.keras.python.keras import regularizers
from tensorflow.contrib.keras.python.keras.engine import InputSpec
from tensorflow.contrib.keras.python.keras.engine import Layer
from tensorflow.contrib.keras.python.keras.utils.generic_utils import get_custom_objects 
from tensorflow.contrib.keras.python.keras.utils import conv_utils

from tensorflow.python.layers import base
from tensorflow.python.layers import utils
from tensorflow.python.layers import convolutional as tf_convolutional_layers

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import nn

import numpy as np

# pylint: disable=redefined-builtin,line-too-long
def separable_conv2d_tf_nn(input,
                     depthwise_filter,
                     pointwise_filter,
                     strides,
                     padding,
                     rate=None,
                     name=None,
                     data_format=None):
  """2-D convolution with separable filters.
  Performs a depthwise convolution that acts separately on channels followed by
  a pointwise convolution that mixes channels.  Note that this is separability
  between dimensions `[1, 2]` and `3`, not spatial separability between
  dimensions `1` and `2`.
  In detail,
      output[b, i, j, k] = sum_{di, dj, q, r]
          input[b, strides[1] * i + di, strides[2] * j + dj, q] *
          depthwise_filter[di, dj, q, r] *
          pointwise_filter[0, 0, q * channel_multiplier + r, k]
  `strides` controls the strides for the depthwise convolution only, since
  the pointwise convolution has implicit strides of `[1, 1, 1, 1]`.  Must have
  `strides[0] = strides[3] = 1`.  For the most common case of the same
  horizontal and vertical strides, `strides = [1, stride, stride, 1]`.
  If any value in `rate` is greater than 1, we perform atrous depthwise
  convolution, in which case all values in the `strides` tensor must be equal
  to 1.
  Args:
    input: 4-D `Tensor` with shape according to `data_format`.
    depthwise_filter: 4-D `Tensor` with shape
      `[filter_height, filter_width, in_channels, channel_multiplier]`.
      Contains `in_channels` convolutional filters of depth 1.
    pointwise_filter: 4-D `Tensor` with shape
      `[1, 1, channel_multiplier * in_channels, out_channels]`.  Pointwise
      filter to mix channels after `depthwise_filter` has convolved spatially.
    strides: 1-D of size 4.  The strides for the depthwise convolution for
      each dimension of `input`.
    padding: A string, either `'VALID'` or `'SAME'`.  The padding algorithm.
      See the @{tf.nn.convolution$comment here}
    rate: 1-D of size 2. The dilation rate in which we sample input values
      across the `height` and `width` dimensions in atrous convolution. If it is
      greater than 1, then all values of strides must be 1.
    name: A name for this operation (optional).
    data_format: The data format for input. Either "NHWC" (default) or "NCHW".
  Returns:
    A 4-D `Tensor` with shape according to 'data_format'. For
      example, with data_format="NHWC", shape is [batch, out_height,
      out_width, out_channels].
  Raises:
    ValueError: If channel_multiplier * in_channels > out_channels,
      which means that the separable convolution is overparameterized.
  """
  with ops.name_scope(name, "separable_conv2d",
                      [input, depthwise_filter, pointwise_filter]) as name:
    input = ops.convert_to_tensor(input, name="tensor_in")
    depthwise_filter = ops.convert_to_tensor(
        depthwise_filter, name="depthwise_filter")
    pointwise_filter = ops.convert_to_tensor(
        pointwise_filter, name="pointwise_filter")

    pointwise_filter_shape = pointwise_filter.get_shape().with_rank(4)
    pointwise_filter_shape[0].assert_is_compatible_with(1)
    pointwise_filter_shape[1].assert_is_compatible_with(1)

    channel_multiplier = depthwise_filter.get_shape().with_rank(4)[3]
    if data_format and data_format == "NCHW":
      in_channels = input.get_shape().with_rank(4)[1]
    else:
      in_channels = input.get_shape().with_rank(4)[3]

    out_channels = pointwise_filter_shape[3]

    if rate is None:
      rate = [1, 1]

    # If any of channel numbers is unknown, then the comparison below returns
    # None. See TensorShape.__gt__().
    #if channel_multiplier * in_channels > out_channels:
    #  raise ValueError("Refusing to perform an overparameterized separable "
    #                   "convolution: channel_multiplier * in_channels = "
    #                   "%d * %d = %d > %d = out_channels" %
    #                   (channel_multiplier, in_channels,
    #                    channel_multiplier * in_channels, out_channels))

    # The layout of the ops in the graph are expected to be as follows:
    # depthwise_conv2d  // Conv2D op corresponding to native deptwise conv.
    # separable_conv2d  // Conv2D op corresponding to the pointwise conv.

    def op(input_converted, _, padding):
      return nn_ops.depthwise_conv2d_native(
          input=input_converted,
          filter=depthwise_filter,
          strides=strides,
          padding=padding,
          data_format=data_format,
          name="depthwise")

    depthwise = nn_ops.with_space_to_batch(
        input=input,
        filter_shape=array_ops.shape(depthwise_filter),
        dilation_rate=rate,
        padding=padding,
        data_format=data_format,
        op=op)

    return nn_ops.conv2d(
        depthwise,
        pointwise_filter, [1, 1, 1, 1],
        padding="VALID",
        data_format=data_format,
        name=name)


def separable_conv2d_keras_backend(x,
                     depthwise_kernel,
                     pointwise_kernel,
                     strides=(1, 1),
                     padding='valid',
                     data_format=None,
                     dilation_rate=(1, 1)):
  """2D convolution with separable filters.
  Arguments:
      x: input tensor
      depthwise_kernel: convolution kernel for the depthwise convolution.
      pointwise_kernel: kernel for the 1x1 convolution.
      strides: strides tuple (length 2).
      padding: padding mode, "valid" or "same".
      data_format: data format, "channels_first" or "channels_last".
      dilation_rate: tuple of integers,
          dilation rates for the separable convolution.
  Returns:
      Output tensor.
  Raises:
      ValueError: if `data_format` is neither `channels_last` or
      `channels_first`.
  """
  if data_format is None:
    data_format = image_data_format()
  if data_format not in {'channels_first', 'channels_last'}:
    raise ValueError('Unknown data_format ' + str(data_format))

  x = _preprocess_conv2d_input(x, data_format)
  padding = _preprocess_padding(padding)
  strides = (1,) + strides + (1,)

  x = separable_conv2d_tf_nn(
      x,
      depthwise_kernel,
      pointwise_kernel,
      strides=strides,
      padding=padding,
      rate=dilation_rate)
  return _postprocess_conv2d_output(x, data_format)


class SeparableConv2DTfLayers(tf_convolutional_layers.Conv2D):
  """Depthwise separable 2D convolution.
  This layer performs a depthwise convolution that acts separately on
  channels, followed by a pointwise convolution that mixes channels.
  If `use_bias` is True and a bias initializer is provided,
  it adds a bias vector to the output.
  It then optionally applies an activation function to produce the final output.
  Arguments:
    filters: Integer, the dimensionality of the output space (i.e. the number
      of filters in the convolution).
    kernel_size: A tuple or list of 2 integers specifying the spatial
      dimensions of of the filters. Can be a single integer to specify the same
      value for all spatial dimensions.
    strides: A tuple or list of 2 positive integers specifying the strides
      of the convolution. Can be a single integer to specify the same value for
      all spatial dimensions.
      Specifying any `stride` value != 1 is incompatible with specifying
      any `dilation_rate` value != 1.
    padding: One of `"valid"` or `"same"` (case-insensitive).
    data_format: A string, one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch, height, width, channels)` while `channels_first` corresponds to
      inputs with shape `(batch, channels, height, width)`.
    dilation_rate: An integer or tuple/list of 2 integers, specifying
      the dilation rate to use for dilated convolution.
      Can be a single integer to specify the same value for
      all spatial dimensions.
      Currently, specifying any `dilation_rate` value != 1 is
      incompatible with specifying any stride value != 1.
    depth_multiplier: The number of depthwise convolution output channels for
      each input channel. The total number of depthwise convolution output
      channels will be equal to `num_filters_in * depth_multiplier`.
    activation: Activation function. Set it to None to maintain a
      linear activation.
    use_bias: Boolean, whether the layer uses a bias.
    depthwise_initializer: An initializer for the depthwise convolution kernel.
    pointwise_initializer: An initializer for the pointwise convolution kernel.
    bias_initializer: An initializer for the bias vector. If None, no bias will
      be applied.
    depthwise_regularizer: Optional regularizer for the depthwise
      convolution kernel.
    pointwise_regularizer: Optional regularizer for the pointwise
      convolution kernel.
    bias_regularizer: Optional regularizer for the bias vector.
    activity_regularizer: Regularizer function for the output.
    trainable: Boolean, if `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    name: A string, the name of the layer.
  """

  def __init__(self, filters,
               kernel_size,
               strides=(1, 1),
               padding='valid',
               data_format='channels_last',
               dilation_rate=(1, 1),
               depth_multiplier=1,
               activation=None,
               use_bias=True,
               depthwise_initializer=None,
               pointwise_initializer=None,
               bias_initializer=init_ops.zeros_initializer(),
               depthwise_regularizer=None,
               pointwise_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               trainable=True,
               name=None,
               **kwargs):
    super(SeparableConv2DTfLayers, self).__init__(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        activation=activation,
        use_bias=use_bias,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        trainable=trainable,
        name=name,
        **kwargs)
    self.depth_multiplier = depth_multiplier
    self.depthwise_initializer = depthwise_initializer
    self.pointwise_initializer = pointwise_initializer
    self.depthwise_regularizer = depthwise_regularizer
    self.pointwise_regularizer = pointwise_regularizer

  def build(self, input_shape):
    if len(input_shape) < 4:
      raise ValueError('Inputs to `SeparableConv2D` should have rank 4. '
                       'Received input shape:', str(input_shape))
    if self.data_format == 'channels_first':
      channel_axis = 1
    else:
      channel_axis = 3
    if input_shape[channel_axis] is None:
      raise ValueError('The channel dimension of the inputs to '
                       '`SeparableConv2D` '
                       'should be defined. Found `None`.')
    input_dim = int(input_shape[channel_axis])
    self.input_spec = base.InputSpec(ndim=4, axes={channel_axis: input_dim})
    depthwise_kernel_shape = (self.kernel_size[0],
                              self.kernel_size[1],
                              input_dim,
                              self.depth_multiplier)
    pointwise_kernel_shape = (1, 1,
                              self.depth_multiplier * input_dim,
                              self.filters)

    self.depthwise_kernel = self.add_variable(
        name='depthwise_kernel',
        shape=depthwise_kernel_shape,
        initializer=self.depthwise_initializer,
        regularizer=self.depthwise_regularizer,
        trainable=True,
        dtype=self.dtype)
    self.pointwise_kernel = self.add_variable(
        name='pointwise_kernel',
        shape=pointwise_kernel_shape,
        initializer=self.pointwise_initializer,
        regularizer=self.pointwise_regularizer,
        trainable=True,
        dtype=self.dtype)
    if self.use_bias:
      self.bias = self.add_variable(name='bias',
                                    shape=(self.filters,),
                                    initializer=self.bias_initializer,
                                    regularizer=self.bias_regularizer,
                                    trainable=True,
                                    dtype=self.dtype)
    else:
      self.bias = None
    self.built = True

  def call(self, inputs):
    if self.data_format == 'channels_first':
      # Reshape to channels last
      inputs = array_ops.transpose(inputs, (0, 2, 3, 1))

    # Apply the actual ops.
    outputs = separable_conv2d_tf_nn(
        inputs,
        self.depthwise_kernel,
        self.pointwise_kernel,
        strides=(1,) + self.strides + (1,),
        padding=self.padding.upper(),
        rate=self.dilation_rate)

    if self.data_format == 'channels_first':
      # Reshape to channels first
      outputs = array_ops.transpose(outputs, (0, 3, 1, 2))

    if self.bias is not None:
      outputs = nn.bias_add(
          outputs,
          self.bias,
          data_format=utils.convert_data_format(self.data_format, ndim=4))

    if self.activation is not None:
      return self.activation(outputs)
    return outputs

  def _compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    if self.data_format == 'channels_first':
      rows = input_shape[2]
      cols = input_shape[3]
    else:
      rows = input_shape[1]
      cols = input_shape[2]

    rows = utils.conv_output_length(rows, self.kernel_size[0],
                                    self.padding, self.strides[0])
    cols = utils.conv_output_length(cols, self.kernel_size[1],
                                    self.padding, self.strides[1])
    if self.data_format == 'channels_first':
      return tensor_shape.TensorShape(
          [input_shape[0], self.filters, rows, cols])
    else:
      return tensor_shape.TensorShape(
    [input_shape[0], rows, cols, self.filters])



class SeparableConv2DKeras(SeparableConv2DTfLayers, Layer):
  """Depthwise separable 2D convolution.
  Separable convolutions consist in first performing
  a depthwise spatial convolution
  (which acts on each input channel separately)
  followed by a pointwise convolution which mixes together the resulting
  output channels. The `depth_multiplier` argument controls how many
  output channels are generated per input channel in the depthwise step.
  Intuitively, separable convolutions can be understood as
  a way to factorize a convolution kernel into two smaller kernels,
  or as an extreme version of an Inception block.
  Arguments:
      filters: Integer, the dimensionality of the output space
          (i.e. the number output of filters in the convolution).
      kernel_size: An integer or tuple/list of 2 integers, specifying the
          width and height of the 2D convolution window.
          Can be a single integer to specify the same value for
          all spatial dimensions.
      strides: An integer or tuple/list of 2 integers,
          specifying the strides of the convolution along the width and height.
          Can be a single integer to specify the same value for
          all spatial dimensions.
          Specifying any stride value != 1 is incompatible with specifying
          any `dilation_rate` value != 1.
      padding: one of `"valid"` or `"same"` (case-insensitive).
      data_format: A string,
          one of `channels_last` (default) or `channels_first`.
          The ordering of the dimensions in the inputs.
          `channels_last` corresponds to inputs with shape
          `(batch, height, width, channels)` while `channels_first`
          corresponds to inputs with shape
          `(batch, channels, height, width)`.
          It defaults to the `image_data_format` value found in your
          Keras config file at `~/.keras/keras.json`.
          If you never set it, then it will be "channels_last".
      depth_multiplier: The number of depthwise convolution output channels
          for each input channel.
          The total number of depthwise convolution output
          channels will be equal to `filterss_in * depth_multiplier`.
      activation: Activation function to use.
          If you don't specify anything, no activation is applied
          (ie. "linear" activation: `a(x) = x`).
      use_bias: Boolean, whether the layer uses a bias vector.
      depthwise_initializer: Initializer for the depthwise kernel matrix.
      pointwise_initializer: Initializer for the pointwise kernel matrix.
      bias_initializer: Initializer for the bias vector.
      depthwise_regularizer: Regularizer function applied to
          the depthwise kernel matrix.
      pointwise_regularizer: Regularizer function applied to
          the depthwise kernel matrix.
      bias_regularizer: Regularizer function applied to the bias vector.
      activity_regularizer: Regularizer function applied to
          the output of the layer (its "activation")..
      depthwise_constraint: Constraint function applied to
          the depthwise kernel matrix.
      pointwise_constraint: Constraint function applied to
          the pointwise kernel matrix.
      bias_constraint: Constraint function applied to the bias vector.
  Input shape:
      4D tensor with shape:
      `(batch, channels, rows, cols)` if data_format='channels_first'
      or 4D tensor with shape:
      `(batch, rows, cols, channels)` if data_format='channels_last'.
  Output shape:
      4D tensor with shape:
      `(batch, filters, new_rows, new_cols)` if data_format='channels_first'
      or 4D tensor with shape:
      `(batch, new_rows, new_cols, filters)` if data_format='channels_last'.
      `rows` and `cols` values might have changed due to padding.
  """

  def __init__(self,
               filters,
               kernel_size,
               strides=(1, 1),
               padding='valid',
               data_format=None,
               depth_multiplier=1,
               activation=None,
               use_bias=True,
               depthwise_initializer='glorot_uniform',
               pointwise_initializer='glorot_uniform',
               bias_initializer='zeros',
               depthwise_regularizer=None,
               pointwise_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               depthwise_constraint=None,
               pointwise_constraint=None,
               bias_constraint=None,
               **kwargs):
    if data_format is None:
      data_format = K.image_data_format()
    super(SeparableConv2DKeras, self).__init__(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        activation=activations.get(activation),
        use_bias=use_bias,
        depthwise_initializer=initializers.get(depthwise_initializer),
        pointwise_initializer=initializers.get(pointwise_initializer),
        bias_initializer=initializers.get(bias_initializer),
        depthwise_regularizer=regularizers.get(depthwise_regularizer),
        pointwise_regularizer=regularizers.get(pointwise_regularizer),
        bias_regularizer=regularizers.get(bias_regularizer),
        activity_regularizer=regularizers.get(activity_regularizer),
        **kwargs)
    # TODO(fchollet): move weight constraint support to core layers.
    self.depthwise_constraint = constraints.get(depthwise_constraint)
    self.pointwise_constraint = constraints.get(pointwise_constraint)
    self.bias_constraint = constraints.get(bias_constraint)

  def build(self, input_shape):
    super(SeparableConv2DKeras, self).build(input_shape)
    # TODO(fchollet): move weight constraint support to core layers.
    if self.depthwise_constraint:
      self.constraints[self.depthwise_kernel] = self.depthwise_constraint
    if self.pointwise_constraint:
      self.constraints[self.pointwise_kernel] = self.pointwise_constraint
    if self.use_bias and self.bias_constraint:
      self.constraints[self.bias] = self.bias_constraint

  def get_config(self):
    config = {
        'filters': self.filters,
        'kernel_size': self.kernel_size,
        'strides': self.strides,
        'padding': self.padding,
        'data_format': self.data_format,
        'activation': activations.serialize(self.activation),
        'use_bias': self.use_bias,
        'depthwise_initializer': initializers.serialize(
            self.depthwise_initializer),
        'pointwise_initializer': initializers.serialize(
            self.pointwise_initializer),
        'bias_initializer': initializers.serialize(self.bias_initializer),
        'depthwise_regularizer': regularizers.serialize(
            self.depthwise_regularizer),
        'pointwise_regularizer': regularizers.serialize(
            self.pointwise_regularizer),
        'bias_regularizer': regularizers.serialize(self.bias_regularizer),
        'activity_regularizer':
            regularizers.serialize(self.activity_regularizer),
        'depthwise_constraint': constraints.serialize(
            self.depthwise_constraint),
        'pointwise_constraint': constraints.serialize(
            self.pointwise_constraint),
        'bias_constraint': constraints.serialize(self.bias_constraint)
    }
    base_config = super(SeparableConv2DKeras, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


def resize_images_bilinear(x, height_factor, width_factor, data_format):
  """Resizes the images contained in a 4D tensor.
  Arguments:
      x: Tensor or variable to resize.
      height_factor: Positive integer.
      width_factor: Positive integer.
      data_format: One of `"channels_first"`, `"channels_last"`.
  Returns:
      A tensor.
  Raises:
      ValueError: if `data_format` is neither
          `channels_last` or `channels_first`.
  """
  if data_format == 'channels_first':
    original_shape = K.int_shape(x)
    new_shape = array_ops.shape(x)[2:]
    new_shape *= constant_op.constant(
        np.array([height_factor, width_factor]).astype('int32'))
    x = permute_dimensions(x, [0, 2, 3, 1])
    x = image_ops.resize_bilinear(x, new_shape)
    x = permute_dimensions(x, [0, 3, 1, 2])
    x.set_shape((None, None, original_shape[2] * height_factor
                 if original_shape[2] is not None else None,
                 original_shape[3] * width_factor
                 if original_shape[3] is not None else None))
    return x
  elif data_format == 'channels_last':
    original_shape = K.int_shape(x)
    new_shape = array_ops.shape(x)[1:3]
    new_shape *= constant_op.constant(
        np.array([height_factor, width_factor]).astype('int32'))
    x = image_ops.resize_bilinear(x, new_shape)
    x.set_shape((None, original_shape[1] * height_factor
                 if original_shape[1] is not None else None,
                 original_shape[2] * width_factor
                 if original_shape[2] is not None else None, None))
    return x
  else:
    raise ValueError('Invalid data_format:', data_format)


class BilinearUpSampling2D(Layer):
  """Upsampling layer for 2D inputs.
  Repeats the rows and columns of the data
  by size[0] and size[1] respectively.
  Arguments:
      size: int, or tuple of 2 integers.
          The upsampling factors for rows and columns.
      data_format: A string,
          one of `channels_last` (default) or `channels_first`.
          The ordering of the dimensions in the inputs.
          `channels_last` corresponds to inputs with shape
          `(batch, height, width, channels)` while `channels_first`
          corresponds to inputs with shape
          `(batch, channels, height, width)`.
          It defaults to the `image_data_format` value found in your
          Keras config file at `~/.keras/keras.json`.
          If you never set it, then it will be "channels_last".
  Input shape:
      4D tensor with shape:
      - If `data_format` is `"channels_last"`:
          `(batch, rows, cols, channels)`
      - If `data_format` is `"channels_first"`:
          `(batch, channels, rows, cols)`
  Output shape:
      4D tensor with shape:
      - If `data_format` is `"channels_last"`:
          `(batch, upsampled_rows, upsampled_cols, channels)`
      - If `data_format` is `"channels_first"`:
          `(batch, channels, upsampled_rows, upsampled_cols)`
  """

  def __init__(self, size=(2, 2), data_format=None, **kwargs):
    super(BilinearUpSampling2D, self).__init__(**kwargs)
    self.data_format = conv_utils.normalize_data_format(data_format)
    self.size = conv_utils.normalize_tuple(size, 2, 'size')
    self.input_spec = InputSpec(ndim=4)

  def _compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    if self.data_format == 'channels_first':
      height = self.size[0] * input_shape[
          2] if input_shape[2] is not None else None
      width = self.size[1] * input_shape[
          3] if input_shape[3] is not None else None
      return tensor_shape.TensorShape(
          [input_shape[0], input_shape[1], height, width])
    else:
      height = self.size[0] * input_shape[
          1] if input_shape[1] is not None else None
      width = self.size[1] * input_shape[
          2] if input_shape[2] is not None else None
      return tensor_shape.TensorShape(
          [input_shape[0], height, width, input_shape[3]])

  def call(self, inputs):
    return resize_images_bilinear(inputs, self.size[0], self.size[1], self.data_format)

  def get_config(self):
    config = {'size': self.size, 'data_format': self.data_format}
    base_config = super(BilinearUpSampling2D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


# add this to custom objects for restoring model save files
get_custom_objects().update({'SeparableConv2DKeras': SeparableConv2DKeras, 'BilinearUpSampling2D':BilinearUpSampling2D})
