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
    ry1 = np.random.randint(0, int(shape[0] * 0.35))
    rx1 = np.random.randint(0, int(shape[1] * 0.35))

    ry2 = np.random.randint(int(shape[0] * 0.65), shape[0])
    rx2 = np.random.randint(int(shape[1] * 0.65), shape[1])

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

        self.file_tuples = list(zip(im_files, mask_files))
        self.n = len(self.file_tuples)

        super(BatchIteratorSimple, self).__init__(self.n, batch_size, shuffle, seed)


    # class function is called next for python 2.x compatability
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
        
        # iterates over the indexes of one batch
        for e, i in enumerate(index_array):
            # TODO implement data loading and preperation logic, make a call to shift_aug at some point. 


        if not self.training:
          return batch_x

        else:
          return batch_x, batch_y


