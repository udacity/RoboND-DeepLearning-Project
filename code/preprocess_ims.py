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

import glob
import os
import shutil
import sys

import numpy as np
from scipy import misc


def make_dir_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_mask_files(image_files):
    is_cam2 = lambda x: x.find('cam2_') != -1
    is_cam3 = lambda x: x.find('cam3_') != -1
    is_cam4 = lambda x: x.find('cam4_') != -1

    cam2 = sorted(list(filter(is_cam2, image_files)))
    cam3 = sorted(list(filter(is_cam3, image_files)))
    cam4 = sorted(list(filter(is_cam4, image_files)))
    return cam2, cam3, cam4


def move_labels(input_folder, output_folder, fold_id):
    files = glob.glob(os.path.join(input_folder, '*', '*.png'))

    output_folder = os.path.join(output_folder, 'masks')
    make_dir_if_not_exist(output_folder)
    cam2, cam3, cam4 = get_mask_files(files) 

    for e,i in enumerate(cam2):
        fname_parts = i.split(os.sep)

        # Thanks @Nitish for these fixes:)
        cam2_base = str(fold_id) + '_' + fname_parts[-3] +'_' + fname_parts[-1]

        fname_parts = cam3[e].split(os.sep)
        cam3_base = str(fold_id) + '_' + fname_parts[-3] +'_' + fname_parts[-1]

        fname_parts = cam4[e].split(os.sep)
        cam4_base = str(fold_id) + '_' + fname_parts[-3] +'_' + fname_parts[-1]

        shutil.copy(i, os.path.join(output_folder,cam2_base))
        shutil.copy(cam3[e], os.path.join(output_folder,cam3_base))
        shutil.copy(cam4[e], os.path.join(output_folder,cam4_base))


def move_png_to_jpeg(input_folder, output_folder, fold_id):
    files = glob.glob(os.path.join(input_folder, '*', '*.png'))
    is_cam1 = lambda x: x.find('cam1_') != -1
    cam1_files = sorted(list(filter(is_cam1, files)))
    output_folder = os.path.join(output_folder, 'images')
    make_dir_if_not_exist(output_folder)

    for i in cam1_files:
        cam1 = misc.imread(i)
        fname_parts = i.split(os.sep)
        cam1_base = str(fold_id) + '_' +fname_parts[-3] + '_' + fname_parts[-1].split('.')[0] + '.jpeg'
        misc.imsave(os.path.join(output_folder, cam1_base), cam1, format='jpeg')


def combine_masks(processed_folder):
    processed_mask_folder = os.path.join(processed_folder, 'masks')
    files = glob.glob(os.path.join(processed_mask_folder, '*.png'))
    cam2, cam3, cam4 = get_mask_files(files)

    for e,i in enumerate(cam2):
        im2 = misc.imread(i)[:,:,0]
        im3 = misc.imread(cam3[e])[:,:,0]
        im4 = misc.imread(cam4[e])[:,:,0]

        stacked = np.stack((im4-1, im2, im3), 2)
        argmin = np.argmin(stacked, axis=-1)
        im = np.stack((argmin==0, argmin==1, argmin==2), 2)

        base_name = os.path.basename(i)
        ind = base_name.find('cam')
        new_fname = base_name[:ind] + 'mask'+ base_name[ind+4:]

        dir_base = str(os.sep).join(i.split(str(os.sep))[:-1])
        misc.imsave(os.path.join(dir_base, new_fname), im)
        os.remove(i)
        os.remove(cam3[e])
        os.remove(cam4[e])


def get_im_data(base_path):
    folds = glob.glob(os.path.join(base_path, '*', '*'))
    indicator_dict = dict()
    
    is_val = lambda x: x.find('validation') != -1
    
    for f in folds:
        files = glob.glob(os.path.join(f, '*','*.png'))
        if len(files) == 0:
            indicator_dict[f] = (False, is_val(f))
        else:
            indicator_dict[f] = (True,  is_val(f))
    return indicator_dict


if __name__ == '__main__':
    raw_data = os.path.join('..', 'data', 'raw_sim_data')
    proc_data = os.path.join('..', 'data', 'processed_sim_data')

    indicator_dict = get_im_data(raw_data) 

    out_val_dir = os.path.join(proc_data, 'validation')
    out_train_dir = os.path.join(proc_data, 'train')

    for e, i in enumerate(indicator_dict.items()):
        # no data in the folder so skip it
        if not i[1][0]:
            continue

        # validation
        if i[1][1]: 
             move_png_to_jpeg(i[0], out_val_dir, e)
             move_labels(i[0], out_val_dir, e)
        # train 
        else:
             move_png_to_jpeg(i[0], out_train_dir, e)
             move_labels(i[0], out_train_dir, e)


    combine_masks(out_val_dir)
    combine_masks(out_train_dir)
