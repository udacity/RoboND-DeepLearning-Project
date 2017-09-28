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

# contains metrics for evaluating the quality of an NN solution
import numpy as np 
from skimage import morphology
from scipy import ndimage as ndi
import glob
import os 
from scipy import misc


def intersection_over_union(y_true, y_pred):
    """Computes the intersection over union of to arrays containing 1's and 0's

    Assumes y_true has converted from real value to binary values. 
    """

    if np.sum(y_true == 1) + np.sum(y_true == 0) != y_true.shape[0]*y_true.shape[1]:
        raise ValueError('Groud truth mask must only contain values from the set {0,1}')

    if np.sum(y_pred == 1) + np.sum(y_pred == 0) != y_pred.shape[0]*y_pred.shape[1]:
        raise ValueError('Segmentation mask must only contain values from the set {0,1}')

    if y_true.ndim != 2:
        if y_true.shape[2] != 1 or y_true.shape[2] != 0:
            raise ValueError('Too many ground truth masks are present')

    if y_pred.ndim != 2:
        if y_pred.shape[2] != 1 or y_pred.shape[2] != 0:
            raise ValueError('too many segmentation masks are present')

    if y_pred.shape != y_true.shape:
        raise ValueError('The dimensions of y_true, and y_pred are not the same')

    intersection = np.sum(y_true * y_pred).astype(np.float)
    union = np.sum(np.clip(y_true + y_pred, 0, 1)).astype(np.float)

    # Alternatively we can return some small value epsilon
    if union == 0:
        # return 1e-10
        return 0

    else:
        return intersection/union # + 1e-10


def jaccard_distance(y_true, y_pred):
    return 1 - intersection_over_union(y_true, y_pred)


def average_squared_distance(y_true, y_pred):
    if y_pred.shape != y_true.shape:
        raise ValueError('The dimensions of y_true, and y_pred are not the same')

    return np.sqrt(np.sum(np.power(y_true - y_pred, 2)))


def average_squared_log_distance(y_true, y_pred):
    if y_pred.shape != y_true.shape:
        raise ValueError('The dimensions of y_true, and y_pred are not the same')
    
    dist = np.abs(y_true-y_pred) 
    return np.sqrt(np.sum(np.power(np.log1p(dist), 2)))


def get_centroid(seg_mask, slices):
    sliced = seg_mask[slices]
    ys, xs = np.where(sliced)
    
    # get the centroid coordinates in the original image
    y_cent = np.round(ys.mean()).astype(np.int) + slices[0].start
    x_cent = np.round(xs.mean()).astype(np.int) + slices[1].start
    return y_cent, x_cent


def find_largest_obj(seg_mask, objs):
    counts = list()
    for slices in objs:
        counts.append((seg_mask[slices]).sum())
    max_id = np.argmax(np.array(counts))
    largest_obj = objs[max_id]
    return largest_obj


def get_centroid_largest_blob(seg_mask):
    labeled_blobs = ndi.label(seg_mask)
    objs = ndi.find_objects(labeled_blobs[0])
    largest_obj = find_largest_obj(seg_mask, objs)
    return np.array(get_centroid(seg_mask, largest_obj))


def score_run_iou(gt_dir, pred_dir):
    gt_files = sorted(glob.glob(os.path.join(gt_dir, 'masks', '*.png')))
    pred_files = sorted(glob.glob(os.path.join(pred_dir, '*.png')))
    ious = [0,0,0]
    n_preds = len(gt_files)
    n_true_pos = 0 
    n_false_neg = 0
    n_false_pos = 0

    for e, gt_file in enumerate(gt_files):
        gt_mask = misc.imread(gt_file).clip(0, 1)
        pred_mask = (misc.imread(pred_files[e]) > 127).astype(np.int)

        if gt_mask.shape[0] != pred_mask.shape[0]:
            gt_mask = misc.imresize(gt_mask, pred_mask.shape)

        for i in range(3):
            ious[i] += intersection_over_union(pred_mask[:,:,i], gt_mask[:,:,i])


        if gt_mask[:,:,2].sum() > 3:
            if pred_mask[:,:, 2].sum() > 3:
                n_true_pos += 1
            else:
                n_false_neg += 1

        else:
            if pred_mask[:, :, 2].sum() > 3:
                n_false_pos += 1

    background = ious[0] / n_preds
    people = ious[1] / n_preds
    hero = ious[2] / n_preds

    print('number of validation samples intersection over the union evaulated on {}'.format(n_preds))
    print('average intersection over union for background is {}'.format(background))
    print('average intersection over union for other people is {}'.format(people))
    print('average intersection over union for the hero is {}'.format(hero))
    print('number true positives: {}, number false positives: {}, number false negatives: {}'.format(n_true_pos, n_false_pos, n_false_neg))
    return n_true_pos, n_false_pos, n_false_neg, hero


def score_run_centroid(gt_dir, pred_dir):
    gt_files = sorted(glob.glob(os.path.join(gt_dir, 'masks', '*.png')))
    pred_files = sorted(glob.glob(os.path.join(pred_dir, '*.png')))

    error1 = 0
    error2 = 0
    n_valid_detections = 0
    n_invalid_detections = 0
    n_missed = 0

    for e, gt_file in enumerate(gt_files):
        gt_mask = misc.imread(gt_file)[:, :, 1].clip(0, 1)
        pred_mask = (misc.imread(pred_files[e])[:, :, 1] > 127).astype(np.int)
        if gt_mask.shape[0] != pred_mask.shape[0]:
            gt_mask = misc.imresize(gt_mask, pred_mask.shape)

        # there target was in the image
        if gt_mask.sum() > 3:
            if pred_mask.sum() > 3:
                gt_centroid = get_centroid_largest_blob(gt_mask)
                pred_centroid = get_centroid_largest_blob(pred_mask)
                error1 += average_squared_distance(pred_centroid, gt_centroid)
                error2 += average_squared_log_distance(pred_centroid, gt_centroid)
                n_valid_detections += 1
            else:
                n_missed += 1

        # the target was not in the image
        else:
            # we got a false positive
            if pred_mask.sum() > 3:
                n_invalid_detections += 1

    n_preds = len(gt_files)
    print('total number of images evaluated on'.format(n_preds))

    print('number of true positives: {}'.format(n_valid_detections))
    print('number of false positives is: {}'.format(n_invalid_detections))
    print('number of false negatives is : {}'.format(n_missed))

    if n_valid_detections > 0:
        print('The following two metrics are only computed on examples where a valid centroid was detected')
        print('average squared pixel distance error {}'.format(error1 / n_valid_detections))
        print('average squared log pixel distance error {}'.format(error2 / n_valid_detections))

