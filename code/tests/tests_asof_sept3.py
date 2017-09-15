#!/usr/bin/env python

import sys
import base64
import math
import numpy as np
from io import BytesIO
from PIL import Image
from scipy import misc

from tensorflow.contrib.keras.python.keras import optimizers
from tensorflow.contrib.keras.python import keras

sys.path.append('..')
from utils import ray_casting
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

def to_radians(deg_ang):
    return deg_ang * (math.pi/180)

# current debugging function
def on_sensor_data(data):
    #print('Sensor frame received')
    rgb_image = Image.open(BytesIO(base64.b64decode(data['rgb_image'])))
    rgb_image = Image.open(BytesIO(base64.b64decode(data['rgb_image'])))
    rgb_image = np.asarray(rgb_image)
    
    if rgb_image.shape != (256,256,3):
        print('image shape not 256, 256, 3')
        return None
    
    # to save the received rgb images
    # np.save(str(np.random.randint(9999)), rgb_image)
    
    # only draw boxes every few seconds
    if np.random.randint(45) > 41:
       
        # cast a ray directly at the center of the image
        # once this works we would need to vary this so that the centroid is not in the center of the image
        # some errors will be suppressed by the symetry of this pixel choice, which is what I want at this moment. 
        ray = ray_casting.cast_ray(data, [128, 128])
        
        # gimbal pose and quad pose: I am pretty sure they are in the world reference frame
        # TODO remove this when sign flip on y is fixed in unity sim
        gimbal_pose = tmpfix_position(list(map(float, data['gimbal_pose'].split(','))))

        # TODO remove this when sign flip on y is fixed in unity sim
        quad_pose = tmpfix_position(list(map(float, data['pose'].split(','))))



        # the sim world has a static reference frame, the angle below is the angle between the world frame and the sensor frame
        rot = transformations.euler_matrix(to_radians(gimbal_pose[3]),
                                           to_radians(gimbal_pose[4]),
                                           -to_radians(gimbal_pose[5]))[:3,:3]

        # this is the identity rotation. The conventions of transformations, is w,x,y,z, and for unity x,y,z,w
        gimbal_cam_rot = transformations.quaternion_matrix((1,0,0,0))[:3,:3]  
        
        # rotate the ray coordinates, so the ray is in the world frame. I think the order may matter here. 
        print('ray unrotated', ray)
        ray = np.dot(gimbal_cam_rot, ray)        
        ray = np.dot(rot, np.array(ray))
        print('ray rotated', ray)

        # print some more debug data
         
        euler = gimbal_pose[3:]
        quaternion = transformations.quaternion_from_euler(euler[0], euler[1], euler[2]) 
        print('gimbal_rotation_quat', quaternion)
        print('gimbal_pose_received', data['gimbal_pose'])
        print('gimbal_pose_fixed', gimbal_pose)

        euler = quad_pose[3:]
        quaternion = transformations.quaternion_from_euler(euler[0], euler[1], euler[2]) 
        print('quad_rotation_quat', quaternion)
        print('quad_position_received', data['pose'])
        print('quad_pose_fixed', quad_pose)
        
        # Flip the sign again so that that the pose received set by the sim is correct
        # TODO remove the call to tmpfix when sign flip on y is fixed in unity sim
        marker_pos = tmpfix_position((gimbal_pose[:3] + ray).tolist() + [0,0,0]).tolist()
        
        marker_rot = [0,0,0]
        quaternion = transformations.quaternion_from_euler(*marker_rot) 
        print('marker_rotation_quat', quaternion, '\n\n')

        # prepare the marker message and send 
        marker_msg = sio_msgs.create_box_marker_msg(np.random.randint(99999), marker_pos)
        socketIO.emit('create_box_marker', marker_msg)

def on_disconnect():
    print('disconnect')

def on_connect():
    print('connect')

def on_reconnect():
    print('reconnect')

if __name__ == '__main__':
    socketIO = SocketIO('localhost', 4567, LoggingNamespace)
    socketIO.on('connect', on_connect)
    socketIO.on('disconnect', on_disconnect)
    socketIO.on('reconnect', on_reconnect)
    socketIO.on('sensor_data', on_sensor_data)
    socketIO.wait(seconds=100000000)
