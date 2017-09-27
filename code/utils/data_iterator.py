# Copyright (c) 2017, Udacity
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# The views and conclusions contained in the software and documentation are those
# of the authors and should not be interpreted as representing official policies,
# either expressed or implied, of the FreeBSD Project

# Author: Devin Anzelmo

import os
from glob import glob
from scipy import misc

import numpy as np
from tensorflow.contrib.keras.python.keras.preprocessing.image import Iterator
from tensorflow.contrib.keras.python.keras import backend as K


def preprocess_input(x):
    x = x/255.
    x = x-0.5
    x = x*2.
    return x

def get_patches(image, image_mask):
    shape = image.shape
    ry1 = np.random.randint(0, int(shape[0] * 0.15))
    rx1 = np.random.randint(0, int(shape[1] * 0.15))

    ry2 = np.random.randint(int(shape[0] * 0.85), shape[0])
    rx2 = np.random.randint(int(shape[1] * 0.85), shape[1])

    small_im = image[ry1: np.clip((ry1+ry2), 0, shape[0]),
                     rx1: np.clip((rx1+rx2), 0, shape[1]),
                     :]

    small_mask = image_mask[ry1: np.clip((ry1+ry2), 0, shape[0]),
                            rx1: np.clip((rx1+rx2), 0, shape[1]),
                            :]

    return small_im, small_mask


def shift_and_pad_augmentation(image, image_mask):
    shape = image.shape
    new_im = np.zeros(shape)
    new_mask = np.zeros(image_mask.shape)
    new_mask[:,:,0] = 1

    im_patch, mask_patch = get_patches(image, image_mask)
    patch_shape = im_patch.shape

    ul_y = np.random.randint(0, shape[0]-patch_shape[0])
    ul_x = np.random.randint(0, shape[1]-patch_shape[1])

    new_im[ul_y:ul_y+patch_shape[0],
           ul_x:ul_x+patch_shape[1],
           :] = im_patch

    new_mask[ul_y:ul_y+patch_shape[0],
             ul_x:ul_x+patch_shape[1],
             :] = mask_patch

    return new_im, new_mask


class BatchIteratorSimple(Iterator):
    def __init__(self, data_folder, batch_size, image_shape,
            num_classes=3, training=True, shuffle=True, seed=None, shift_aug=False):

        self.num_classes = num_classes
        self.shift_aug = shift_aug
        self.data_folder = data_folder
        self.batch_size = batch_size
        self.training = training
        self.image_shape = tuple(image_shape)

        im_files = sorted(glob(os.path.join(data_folder, 'images', '*.jpeg')))
        mask_files = sorted(glob(os.path.join(data_folder, 'masks', '*.png')))

        if len(im_files) == 0:
            raise ValueError('No image files found, check your image diractories')

        if len(mask_files) == 0:
            raise ValueError('No mask files found, check your mask directories')

        self.file_tuples = list(zip(im_files, mask_files))
        self.n = len(self.file_tuples)

        super(BatchIteratorSimple, self).__init__(self.n, batch_size, shuffle, seed)


    def next(self):
        """For python 2.x.
        Returns:
            The next batch.
        """
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch.
        with self.lock:
          index_array, current_index, current_batch_size = next(
              self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        
        batch_x = np.zeros((current_batch_size,) + self.image_shape, dtype=K.floatx())


        if self.training:
            batch_y = np.zeros(
                    (current_batch_size,) + self.image_shape[:2] + (self.num_classes,),
                    dtype=K.floatx())

        for e, i in enumerate(index_array):
            # load labels of the person in focus
            file_tuple = self.file_tuples[i]
            image = misc.imread(file_tuple[0])

            if image.shape != self.image_shape:
                image = misc.imresize(image, self.image_shape)

            if not self.training:
                image = preprocess_input(image.astype(np.float32))
                batch_x[e,:,:,:] = image
                continue

            else:
                gt_image = misc.imread(file_tuple[1]).clip(0,1) 
                if gt_image.shape[0] != self.image_shape[0]:
                    gt_image = misc.imresize(gt_image, self.image_shape)

                #if self.shift_aug:
                #    image, gt_image = shift_and_pad_augmentation(image, gt_image)

                image = preprocess_input(image.astype(np.float32))
                batch_x[e,:,:,:] = image
                batch_y[e,:,:,:] = gt_image


        if not self.training:
            return batch_x

        else:
            return batch_x, batch_y
