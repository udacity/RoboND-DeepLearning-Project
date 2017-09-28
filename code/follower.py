# Copyright (c) 2017, Ele ctric Movement
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

# Author: Brandon Kinman

import eventlet.wsgi
import socketio
import time

from flask import Flask
from threading import Thread

# Needs to be sorted through
import argparse
import base64
import math
import numpy as np
import os
import tensorflow as tf
from PIL import Image
from io import BytesIO
from scipy import misc

from transforms3d.euler import euler2mat, mat2euler
from tensorflow.contrib.keras.python import keras
from utils import separable_conv2d

from utils import data_iterator
from utils import visualization
from utils import scoring_utils
from utils import sio_msgs
from utils import model_tools

import time

import signal
import sys


# Create socketio server and Flask app
sio = socketio.Server()
app = Flask(__name__)


def to_radians(deg_ang):
    return deg_ang * (math.pi / 180)


# Functions for 2D->3D transformation
def get_depth_image(data):
    pimg = Image.open(BytesIO(base64.b64decode(data)))
    img_array = np.array(pimg)
    return img_array


def get_xyz_from_image(u, v, depth, im_hw):
    cx = im_hw//2
    cy = im_hw//2
    fx = im_hw
    fy = im_hw
    x = (u-cx)*depth/fx
    y = (v-cy)*depth/fy
    return [depth,-x,y]


def get_ros_pose(data):
    s_pose = data.split(",")
    ros_pose = [float(i) for i in s_pose]
    for i in range(3,6):
        if ros_pose[i]<-180: ros_pose[i] = 360 + ros_pose[i]
    return ros_pose


def get_unity_pose_from_ros(data):
    unity_point = [-data[1],data[2],data[0]]
    return unity_point


class Follower(object):
    def __init__(self, image_hw, model, pred_viz_enabled = False, queue=None):
       
        self.queue = queue
        self.model = model
        self.image_hw = image_hw
        self.last_time_saved = time.time()
        self.num_no_see = 0
        self.pred_viz_enabled = pred_viz_enabled
        self.target_found = False


    def on_sensor_frame(self, data):
        rgb_image = Image.open(BytesIO(base64.b64decode(data['rgb_image'])))
        rgb_image = np.asarray(rgb_image)

        if rgb_image.shape != (256, 256, 3):
            print('image shape not 256, 256, 3')
            return None

        if rgb_image.shape[0] != self.image_hw:
            rgb_image = misc.imresize(rgb_image, (self.image_hw, self.image_hw, 3))

        rgb_image = data_iterator.preprocess_input(rgb_image)
        pred = np.squeeze(model.predict(np.expand_dims(rgb_image, 0)))

        if self.pred_viz_enabled:
            self.queue.put([rgb_image, pred])

        target_mask = pred[:, :, 2] > 0.5
        # reduce the number of false positives by requiring more pixels to be identified as containing the target
        if target_mask.sum() > 10:
            centroid = scoring_utils.get_centroid_largest_blob(target_mask)

            # scale the centroid from the nn image size to the original image size
            centroid = centroid.astype(np.int).tolist()

            # Obtain 3D world point from centroid pixel
            depth_img = get_depth_image(data['depth_image'])

            # Get XYZ coordinates for specific pixel
            pixel_depth = depth_img[centroid[0]][centroid[1]][0]*50/255.0
            point_3d = get_xyz_from_image(centroid[0], centroid[1], pixel_depth, self.image_hw)
            point_3d.append(1)

            # Get cam_pose from sensor_frame (ROS convention)
            cam_pose = get_ros_pose(data['gimbal_pose'])

            # Calculate xyz-world coordinates of the point corresponding to the pixel
            # Transformation Matrix
            R = euler2mat(math.radians(cam_pose[3]), math.radians(cam_pose[4]), math.radians(cam_pose[5]))
            T = np.c_[R, cam_pose[:3]]
            T = np.vstack([T, [0,0,0,1]]) # transformation matrix from world to quad

            # 3D point in ROS coordinates
            ros_point = np.dot(T, point_3d)

            sio.emit('object_detected', {'coords': [ros_point[0], ros_point[1], ros_point[2]]})
            if not self.target_found:
                print('Target found!')
                self.target_found = True
            self.num_no_see = 0

            # Publish Hero Marker
            marker_pos = [ros_point[0],ros_point[1], ros_point[2]] + [0, 0, 0]
            marker_msg = sio_msgs.create_box_marker_msg(np.random.randint(99999), marker_pos)
            sio.emit('create_box_marker', marker_msg)

        elif self.target_found:
            self.num_no_see += 1
            # print(self.num_no_see)

        if self.target_found and self.num_no_see > 6:
            print('Target lost!')
            sio.emit('object_lost', {'data': ''})
            self.target_found = False
            self.num_no_see = 0


######################
# SocketIO Callbacks
######################

@sio.on('sensor_frame')
def sensor_frame(sid, data):
    #global graph
    #with graph.as_default():
    follower.on_sensor_frame(data)


def sio_server():
    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)


if __name__ == '__main__':
    # wrap Flask application with socketio's middleware
    app = socketio.Middleware(sio, app)

    parser = argparse.ArgumentParser()

    parser.add_argument('weight_file',
                        help='The model file to use for inference')


    parser.add_argument('--pred_viz',
                        action='store_true',
                        help='display live overlay visualization with prediction regions')

    args = parser.parse_args()

    model = model_tools.load_network(args.weight_file)
    image_hw = model.layers[0].output_shape[1]

    if args.pred_viz: 
        overlay_plot = visualization.SideBySidePlot('Segmentation Overlay', image_hw)
        queue = overlay_plot.start()
    else:
        queue = None

    follower = Follower(image_hw, model, args.pred_viz, queue)
    # start eventlet server
    
    sio_server()
