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

import argparse
import base64
import math
import numpy as np
import os
import time
from PIL import Image
from io import BytesIO
from scipy import misc

import cv2
from socketIO_client import SocketIO, LoggingNamespace

import make_model
from utils import data_iterator
from utils import preprocess_ims
from utils import visualization


def to_radians(deg_ang):
    return deg_ang * (math.pi / 180)


def overlay_viz(image, pred):
    result = visualization.overlay_predictions(image, pred, None, 0.5, 1)
    result = result.convert('RGB')
    result = np.asarray(result)
    cv2.imshow('result', result[:, :, ::-1].copy())
    cv2.waitKey(1)


class Follower(object):
    def __init__(self, image_save_dir, pred_viz_enabled = False):
        self.image_save_dir = image_save_dir
        self.last_time_saved = time.time()
        self.num_no_see = 0
        self.pred_viz_enabled = pred_viz_enabled
        self.target_found = False

    def on_sensor_data(self, data):
        rgb_image = Image.open(BytesIO(base64.b64decode(data['rgb_image'])))
        rgb_image = np.asarray(rgb_image)

        if rgb_image.shape != (256, 256, 3):
            print('image shape not 256, 256, 3')
            return None

        rgb_image = data_iterator.preprocess_input(rgb_image)
        pred = np.squeeze(model.predict(np.expand_dims(rgb_image, 0)))

        target_mask = pred[:, :, 1] > 0.5

        if self.pred_viz_enabled:
            overlay_viz(rgb_image, pred)

        # reduce the number of false positives by requiring more pixels to be identified as containing the target
        if target_mask.sum() > 10:

            # Temporary move so we only get the overlays with positive identification
            if self.image_save_dir is not None:
                if time.time() - self.last_time_saved > 1:
                    result = visualization.overlay_predictions(rgb_image, pred, None, 0.5, 1)
                    out_file = os.path.join(self.image_save_dir, 'overlay_' + str(time.time()) + '.png')
                    misc.imsave(out_file, result)
                    self.last_time_saved = time.time()

            # centroid = scoring_utils.get_centroid_largest_blob(target_mask)

            # scale the centroid from the nn image size to the original image size
            # centroid = centroid.astype(np.int).tolist()

            # with the depth image, and centroid from prediction we can compute
            # the x,y,z coordinates where the ray intersects an object

            # ray = ray_casting.cast_ray(data, [centroid[1], centroid[0]])
            # pose = np.array(list(map(float, data['gimbal_pose'].split(','))))

            # TODO add rotation of the camera with respect to the gimbal

            # create the rotation matrix to rotate the sensors frame of reference to the world frame
            # rot = transformations.euler_matrix(to_radians(pose[3]),
            #                                   to_radians(pose[4]),
            #                                   to_radians(pose[5]))[:3, :3]

            # rotate array
            # ray = np.dot(rot, np.array(ray))

            # marker_pos = (pose[:3] + ray).tolist() + [0, 0, 0]
            # marker_msg = sio_msgs.create_box_marker_msg(np.random.randint(99999), marker_pos)
            # socketIO.emit('create_box_marker', marker_msg)
            socketIO.emit('object_detected', {'coords': [0, 0, 0]})
            self.target_found = True
            self.num_no_see = 0

        elif self.target_found:
            self.num_no_see += 1

        if self.target_found and self.num_no_see > 8:
            socketIO.emit('object_lost', {'data': ''})
            self.target_found = False
            self.num_no_see = 0


def on_disconnect():
    print('disconnect')


def on_connect():
    print('connect')


def on_reconnect():
    print('reconnect')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('model_file',
                        help='The model file to use for inference')

    parser.add_argument('--pred_viz',
                        action='store_true',
                        help='display live overlay visualization with prediction regions')

    parser.add_argument('--pred_images',
                        help='Save images with prediction overlay, parameters is directory name to save to.')


    args = parser.parse_args()

    model_path = os.path.join('..', 'data', 'weights', args.model_file)
    model = make_model.make_example_model()
    model.load_weights(model_path)

    pred_images_path = None
    if args.pred_images is not None:
        pred_images_path = os.path.join('..', 'data', 'runs', args.pred_images)
        preprocess_ims.make_dir_if_not_exist(pred_images_path)

    follower = Follower(pred_images_path, args.pred_viz)

    socketIO = SocketIO('localhost', 4567, LoggingNamespace)
    socketIO.on('connect', on_connect)
    socketIO.on('disconnect', on_disconnect)
    socketIO.on('reconnect', on_reconnect)
    socketIO.on('sensor_data', follower.on_sensor_data)
    socketIO.wait(seconds=100000000)
