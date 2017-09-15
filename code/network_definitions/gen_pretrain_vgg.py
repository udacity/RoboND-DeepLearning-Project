import shutil
import numpy as np
import os.path
import tensorflow as tf


class FCN8VGG:
    VGG_MEAN = [103.939, 116.779, 123.68]

    def __init__(self, vgg16_npy_path=None):
        self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
        self.wd = 5e-4

    def build(self, rgb, keep_prob):
        with tf.name_scope('Processing'):

            red, green, blue = tf.split(rgb, 3, 3)
            bgr = tf.concat([
                blue - self.VGG_MEAN[0],
                green - self.VGG_MEAN[1],
                red - self.VGG_MEAN[2],
            ], 3)

        self.conv1_1 = self._conv_layer(bgr, "conv1_1")
        self.conv1_2 = self._conv_layer(self.conv1_1, "conv1_2")
        self.pool1 = self._max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self._conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self._conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self._max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self._conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self._conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self._conv_layer(self.conv3_2, "conv3_3")
        self.pool3 = self._max_pool(self.conv3_3, 'pool3')

        tf.identity(self.pool3, 'layer3_out')

        self.conv4_1 = self._conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self._conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self._conv_layer(self.conv4_2, "conv4_3")
        self.pool4 = self._max_pool(self.conv4_3, 'pool4')

        tf.identity(self.pool4, 'layer4_out')

        self.conv5_1 = self._conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self._conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self._conv_layer(self.conv5_2, "conv5_3")
        self.pool5 = self._max_pool(self.conv5_3, 'pool5')

        self.fc6 = self._fc_layer(self.pool5, "fc6")
        self.fc6 = tf.nn.dropout(self.fc6, keep_prob)

        self.fc7 = self._fc_layer(self.fc6, "fc7")
        self.fc7 = tf.nn.dropout(self.fc7, keep_prob)

        tf.identity(self.fc7, 'layer7_out')

    def _max_pool(self, bottom, name):
        pool = tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                              padding='SAME', name=name)
        return pool

    def _conv_layer(self, bottom, name):
        with tf.variable_scope(name) as scope:
            filt = self.get_conv_filter(name)
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            # Add summary to Tensorboard
            return relu

    def _fc_layer(self, bottom, name, num_classes=None,
                  relu=True):
        with tf.variable_scope(name) as scope:
            shape = bottom.get_shape().as_list()

            if name == 'fc6':
                filt = self.get_fc_weight_reshape(name, [7, 7, 512, 4096])
            elif name == 'score_fr':
                name = 'fc8'  # Name of score_fr layer in VGG Model
                filt = self.get_fc_weight_reshape(name, [1, 1, 4096, 1000],
                                                  num_classes=num_classes)
            else:
                filt = self.get_fc_weight_reshape(name, [1, 1, 4096, 4096])

            self._add_wd_and_summary(filt, self.wd, "fc_wlosses")

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            conv_biases = self.get_bias(name, num_classes=num_classes)
            bias = tf.nn.bias_add(conv, conv_biases)

            if relu:
                bias = tf.nn.relu(bias)

            return bias

    def get_conv_filter(self, name):
        init = tf.constant_initializer(value=self.data_dict[name][0],
                                       dtype=tf.float32)
        shape = self.data_dict[name][0].shape
        var = tf.get_variable(name="filter", initializer=init, shape=shape)
        if not tf.get_variable_scope().reuse:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), self.wd,
                                       name='weight_loss')
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,
                                 weight_decay)
        return var

    def get_bias(self, name, num_classes=None):
        bias_wights = self.data_dict[name][1]
        shape = self.data_dict[name][1].shape
        if name == 'fc8':
            bias_wights = self._bias_reshape(bias_wights, shape[0],
                                             num_classes)
            shape = [num_classes]
        init = tf.constant_initializer(value=bias_wights,
                                       dtype=tf.float32)
        var = tf.get_variable(name="biases", initializer=init, shape=shape)
        return var

    def _bias_reshape(self, bweight, num_orig, num_new):
        """ Build bias weights for filter produces with `_summary_reshape`

        """
        n_averaged_elements = num_orig//num_new
        avg_bweight = np.zeros(num_new)
        for i in range(0, num_orig, n_averaged_elements):
            start_idx = i
            end_idx = start_idx + n_averaged_elements
            avg_idx = start_idx//n_averaged_elements
            if avg_idx == num_new:
                break
            avg_bweight[avg_idx] = np.mean(bweight[start_idx:end_idx])
        return avg_bweight

    def _summary_reshape(self, fweight, shape, num_new):
        """ Produce weights for a reduced fully-connected layer.

        FC8 of VGG produces 1000 classes. Most semantic segmentation
        task require much less classes. This reshapes the original weights
        to be used in a fully-convolutional layer which produces num_new
        classes. To archive this the average (mean) of n adjanced classes is
        taken.

        Consider reordering fweight, to perserve semantic meaning of the
        weights.

        Args:
          fweight: original weights
          shape: shape of the desired fully-convolutional layer
          num_new: number of new classes


        Returns:
          Filter weights for `num_new` classes.
        """
        num_orig = shape[3]
        shape[3] = num_new
        assert(num_new < num_orig)
        n_averaged_elements = num_orig//num_new
        avg_fweight = np.zeros(shape)
        for i in range(0, num_orig, n_averaged_elements):
            start_idx = i
            end_idx = start_idx + n_averaged_elements
            avg_idx = start_idx//n_averaged_elements
            if avg_idx == num_new:
                break
            avg_fweight[:, :, :, avg_idx] = np.mean(
                fweight[:, :, :, start_idx:end_idx], axis=3)
        return avg_fweight

    def _add_wd_and_summary(self, var, wd, collection_name=None):
        if collection_name is None:
            collection_name = tf.GraphKeys.REGULARIZATION_LOSSES
        if wd and (not tf.get_variable_scope().reuse):
            weight_decay = tf.multiply(
                tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.add_to_collection(collection_name, weight_decay)
        return var

    def get_fc_weight_reshape(self, name, shape, num_classes=None):
        weights = self.data_dict[name][0]
        weights = weights.reshape(shape)
        if num_classes is not None:
            weights = self._summary_reshape(weights, shape,
                                            num_new=num_classes)
        init = tf.constant_initializer(value=weights,
                                       dtype=tf.float32)
        var = tf.get_variable(name="weights", initializer=init, shape=shape)
        return var


def save_model():
    data_dir = './data'
    export_dir = os.path.join(data_dir, './vgg')

    if os.path.exists(export_dir):
        shutil.rmtree(export_dir)

    builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
    with tf.Session() as sess:
        image_pl = tf.placeholder(tf.float32, [None, None, None, 3], 'image_input')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        # Build VGG
        vgg_fcn = FCN8VGG(vgg16_npy_path=os.path.join(data_dir, 'vgg16.npy'))
        vgg_fcn.build(image_pl, keep_prob)

        sess.run(tf.global_variables_initializer())
        builder.add_meta_graph_and_variables(sess, ['vgg16'], assets_collection=tf.get_collection(tf.GraphKeys.ASSET_FILEPATHS))
        builder.save()


def test_model():
    with tf.Session() as sess:
        tf.saved_model.loader.load(sess, ['vgg16'], './data/vgg')
        graph = tf.get_default_graph()
        image_pl = graph.get_tensor_by_name("image_input:0")
        keep_prob = graph.get_tensor_by_name("keep_prob:0")

        vgg_layer3_out = graph.get_tensor_by_name("layer3_out:0")
        vgg_layer4_out = graph.get_tensor_by_name("layer4_out:0")
        vgg_layer7_out = graph.get_tensor_by_name("layer7_out:0")

        print(sess.run(vgg_layer7_out, {image_pl: np.zeros([1, 1, 1, 3]), keep_prob: 1.0}))
        print('[[[[ 0.53621131  0.55275095  0.65965402 ...,  0.41847846  0.44312999 0.60610819]]]]')

        num_classes = 2
        score_layer7 = tf.layers.conv2d(
            vgg_layer7_out,
            num_classes,
            1,
            padding='same',
            kernel_initializer=tf.truncated_normal_initializer(stddev=1e-2),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-1))

        # Fuse 1
        score_layer4 = tf.layers.conv2d(
            vgg_layer4_out,
            num_classes,
            1,
            padding='same',
            kernel_initializer=tf.truncated_normal_initializer(stddev=1e-3),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-2))
        upscore1 = tf.layers.conv2d_transpose(
            score_layer7,
            num_classes,
            4,
            2,
            'same',
            kernel_initializer=tf.truncated_normal_initializer(stddev=1e-3),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-2))
        fuse1 = tf.add(upscore1, score_layer4)

        # Fuse 2
        score_layer3 = tf.layers.conv2d(
            vgg_layer3_out,
            num_classes,
            1,
            padding='same',
            kernel_initializer=tf.truncated_normal_initializer(stddev=1e-4),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
        upscore2 = tf.layers.conv2d_transpose(
            fuse1,
            num_classes,
            4,
            2,
            'same',
            kernel_initializer=tf.truncated_normal_initializer(stddev=1e-4),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
        fuse2 = tf.add(upscore2, score_layer3)

        # Upsample to original image size
        upscore3 = tf.layers.conv2d_transpose(
            fuse2,
            num_classes,
            16,
            8,
            'same',
            kernel_initializer=tf.truncated_normal_initializer(stddev=1e-5),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4))

        sess.run(tf.global_variables_initializer())
        print(sess.run(upscore3, {image_pl: np.zeros([1, 1, 1, 3]), keep_prob: 1.0}))
        print('[[[[ 0.04574377 -0.17475264]]]]')


if __name__ == '__main__':
    with tf.Graph().as_default():
        save_model()
    with tf.Graph().as_default():
        test_model()
