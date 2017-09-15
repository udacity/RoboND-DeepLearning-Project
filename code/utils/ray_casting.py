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
import math
import numpy as np

from io import BytesIO
from PIL import Image

def cast_ray(sensor_frame, pixel_coords):
    """ Given pixel coordinates, returns (x,y,z) coordinate tuple,
        expressed in the sensor's coordinate frame.

        Functionality is achieved by casting a ray from the camera's
        focal point, through the pixel location on the image plane.
        Depth is determined by indexing to the sensor frame's associated
        depth image. The initial pose of the camera is provided in the
        sensor frame.

        Args:
            sensor_frame (dict): Sensor frame message from the simulator.
            pixel_coords (dict([x,y])): pixel coords in sensor_frame

        Returns:
            (x,y,z): Coordinates of object which was hit.
    """
    z_far = 500.0
    z_near = 0.1
    hres = 256
    vres = 256
    focal_len = z_near
    vertical_fov_deg = 60
    horizontal_fov_deg = 60

    # maximum and minimum value in the depth image
    depth_max = 255.
    depth_min = 0.

    vertical_fov_rad = vertical_fov_deg * math.pi / 180
    horizontal_fov_rad = horizontal_fov_deg * math.pi / 180

    #FYI: +x axis points from origin to the center of the image plane
    #     +y axis points towards the left of the image plane.
    #     +z axis points towards the top of the image plane

    origin = np.array([0, 0, 0], dtype=float)

    # w is a unit vector pointing towards center of image plane from origin
    # (along the +x axis)
    w = np.array([focal_len, 0, 0] - origin, dtype=float)
    w = w / np.linalg.norm(w)

    # u is a unit vector pointing towards the upper right of the image from the upper left
    # (-y according the coordinate convention stated above)
    #vup is simply a gector poininng "upwards"
    vup = np.array([0, 0, 1], dtype=float)
    u = - np.cross(vup, w)

    # v is a unit vector pointing towards the lower left of the image from the upper left
    # (-z according to the coordinate convention stated above)
    v = np.cross(w, u)

    # note, in the above coordinate system, the upper left corner of the
    # image is at (0,0), and the lower right is at (hres-1, vres-1)
    half_height = math.tan(vertical_fov_rad/2)
    half_width = math.tan(horizontal_fov_rad/2)
    upper_left_corner = focal_len*w - half_width * u - half_height * v
    horizontal = 2.0 * half_width * u
    vertical = 2.0 * half_height * v

    #vector representing a single pixel increment along u and v
    u_increment = horizontal / float(hres)
    v_increment = vertical / float(vres)

    # Ray points from the camera origin to the pixel coordinate of interest
    u_idx = pixel_coords[0]
    v_idx = pixel_coords[1]
    ray = upper_left_corner

    ray += u_idx * u_increment + u_increment/2
    ray += v_idx * v_increment + v_increment/2
    ray = ray / np.linalg.norm(ray)

    depth_image = Image.open(BytesIO(base64.b64decode(sensor_frame['depth_image'])))
    depth_image = np.asarray(depth_image)
    
    # scale the depth image between z_near, and z_far
    # f(min_depth) = z_near, because min_depth = 0 
    a = (z_far-z_near)/depth_max
    b = z_near
    
    # f(x) = ax + b
    z = a * (depth_image[u_idx,v_idx,0]) - b

    return z*ray