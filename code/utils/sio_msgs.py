#!/usr/bin/env python

# Copyright (c) 2017, Udacity Inc.
# All rights reserved.

# Author: Brandon Kinman
import base64

from io import BytesIO
from PIL import Image


def create_box_marker_msg(id, pose, dims=(0.7,0.7,2.0), color=(0,0,1,0.1), duration=0.4):
    """ Creates a box marker message to be passed to the simulator.

        Args:
            id (int): Identifier for the box marker.
            pose (list): Pose of the box [x, y, z, roll, pitch, yaw]
            dims (list): Dimensions of the box in meters [height, width, depth]
            color (list): Color and Alpha of the box, with all values as floats
                          ranging from 0-1 [r,g,b,a]
            duration (int): How long the box will exist, -1 means forever

        Returns:
            dict: JSON message to be emitted as socketio event
    """
    msg = {
        'id': id,
        'pose': pose,
        'dimensions': dims,
        'color': color,
        'duration': duration
    }
    return msg

def create_object_detected_msg(position):
    """creates an xyz message of target location"""
    return {'coords':position}


def create_delete_marker_msg(id):
    """ Creates a delete marker message.

        Marker with the given id will be delete, -1 deletes all markers

        Args:
            id (int): Identifier of the marker

        Returns:
            dict: JSON message to be emitted as socketio event
    """
    return {'id': id}
