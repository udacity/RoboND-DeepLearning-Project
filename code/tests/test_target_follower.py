#!/usr/bin/env python

import sys
import base64
import math
import numpy as np
from io import BytesIO
from PIL import Image
from scipy import misc
import time

from tensorflow.contrib.keras.python.keras import optimizers
from tensorflow.contrib.keras.python import keras

sys.path.append('..')
from utils import sio_msgs
from utils import transformations

from socketIO_client import SocketIO, LoggingNamespace

# the y position value seems to have its sign switched
# every time it goes to or from the sim
# TODO remove this patch when the sign switch is fixed in the sim
def tmpfix_position(pose):
    pose = np.array(pose)
    # swap the sign of y
    pose[1] = -pose[1]
    return pose

def tmpfix_position_list(pose):
    pose = np.array(pose)
    # swap the sign of y
    pose[1] = -pose[1]
    return pose.tolist()

def to_radians(deg_ang):
    return deg_ang * (math.pi/180)


def distance(a, b):
     return np.sqrt(np.sum(np.power(a - b, 2)))     


class TargIter(object):
    def __init__(self, targets):
        self.targets = targets
        self.n_targets = len(targets)
        self.current_index = 0

    def update(self):
        pass
    
    def __iter__(self):
        return self

    def __next__(self):
        if self.current_index == self.n_targets: 
            return None

        else:
            ret_val = self.targets[self.current_index]
            self.current_index += 1
            return np.array(ret_val)

class ObjectiveUpdater(object):

    def __init__(self, target_iter):
        self.target_iter = target_iter 
        self.current_target = None
        self.last_pose = None
        self.next_marker_id = 0

    def emit_marker(self, pose):
        msg = sio_msgs.create_box_marker_msg(self.next_marker_id, pose) 
        self.next_marker_id += 1
        print(msg)
        socketIO.emit('create_box_marker', msg)

    def update_target(self, pose): 

        self.last_pose = pose
        self.current_target = next(self.target_iter)

        if self.current_target is None:
            return None

        self.emit_marker(tmpfix_position_list(self.current_target.tolist() + [0,0,0]))
        msg = sio_msgs.create_object_detected_msg(tmpfix_position_list(self.current_target))
        socketIO.emit('object_detected',msg)

    def on_sensor_data(self, data):

        # TODO remove this when sign flip on y is fixed in unity sim
        quad_pose = tmpfix_position(list(map(float, data['pose'].split(','))))


        if self.last_pose is None: 
            self.update_target(quad_pose)

        elif self.current_target is None:
            return None

        elif distance(self.last_pose[:3], self.current_target) < 1.25:
            self.update_target(quad_pose)


        self.last_pose = quad_pose


def on_disconnect():
    print('disconnect')


def on_connect():
    print('connect')


def on_reconnect():
    print('reconnect')


if __name__ == '__main__':
    targ_iter = TargIter([[0, 0, 55], [50,0,55], [50,50,55],[0, 50, 55], [0, 0, 55], [0,0,10]])
    updater = ObjectiveUpdater(targ_iter)
    socketIO = SocketIO('localhost', 4567, LoggingNamespace)
    socketIO.on('connect', on_connect)
    socketIO.on('disconnect', on_disconnect)
    socketIO.on('reconnect', on_reconnect)
    socketIO.on('sensor_data', updater.on_sensor_data)
    socketIO.wait(seconds=100000000)
