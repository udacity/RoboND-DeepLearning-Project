import glob
import numpy as np
import os
import sys
from scipy import misc

from utils import scoring_utils


def score_run(gt_dir, pred_dir):
    gt_files = sorted(glob.glob(os.path.join(gt_dir, 'masks', '*.png')))
    pred_files = sorted(glob.glob(os.path.join(pred_dir, '*.png')))

    iou = 0
    error1 = 0
    error2 = 0
    n_valid_detections = 0
    n_invalid_detections = 0

    for e, gt_file in enumerate(gt_files):
        gt_mask = misc.imread(gt_file)[:, :, 1].clip(0, 1)
        pred_mask = (misc.imread(pred_files[e])[:, :, 1] > 127).astype(np.int)

        iou += scoring_utils.intersection_over_union(pred_mask, gt_mask)
        # only compute centroid distance if it found some object
        if pred_mask.sum() > 3:
            if gt_mask.sum() > 3:
                gt_centroid = scoring_utils.get_centroid_largest_blob(gt_mask)
                pred_centroid = scoring_utils.get_centroid_largest_blob(pred_mask)
                error1 += scoring_utils.average_squared_distance(pred_centroid, gt_centroid)
                error2 += scoring_utils.average_squared_log_distance(pred_centroid, gt_centroid)
                n_valid_detections += 1
            else:
                error1 += 1
                error2 += 1
                n_invalid_detections += 1

    return iou, error1, error2, len(gt_files), n_valid_detections, n_invalid_detections


if __name__ == '__main__':

    if len(sys.argv) != 3:
        raise ValueError('evaluate.py the ground truth folder name and prediction folder name as commandline input')

    gt_folder = sys.argv[1]
    pred_folder = sys.argv[2]

    gt_path = os.path.join('..', 'data', gt_folder)
    pred_path = os.path.join('..', 'data', 'runs', pred_folder)

    (iou, err1, err2, n_preds, n_valid_detections, n_invalid_detections
     ) = score_run(gt_path, pred_path)

    print('average intersection over union {}'.format(iou / n_preds))
    print('number of validation samples evaluated on {}'.format(n_preds))
    print('number of images with target detected: {}'.format(n_valid_detections))
    print('number of images false positives is: {}'.format(n_invalid_detections))
    if n_valid_detections > 0 or n_invalid_detections > 0:
        n_detections = n_valid_detections + n_invalid_detections
        print('average squared pixel distance error {}'.format(err1 / n_detections))
        print('average squared log pixel distance error {}'.format(err2 / n_detections))
