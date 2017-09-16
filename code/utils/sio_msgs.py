#!/usr/bin/env python

# Copyright (c) 2017, Electric Movement
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
