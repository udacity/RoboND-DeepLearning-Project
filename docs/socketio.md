# socketio event definitions #
This project uses [socketio](https://socket.io/) to facilitate communication between QuadSim and Python code.
Following is a list of event definitions and field descriptions that are used. For more information about socketio, check out the official website. The documentation for the python socketio server library can be found 
[here](https://python-socketio.readthedocs.io/en/latest/). Similarly, The documentation for the python socketio client library used in this project can found [here](https://pypi.python.org/pypi/socketIO-client).


## `sensor_frame` event (emitted by QuadSim) ##
These events are published by QuadSim and contain everything needed to allow the quad to detect and follow a detected object in the scene.

Example:
```
{
  "timestamp": "1504814505.23149",
  "rgb_image": "/9j/4AAQS...",
  "pose": "0.0000,0.0000,10.0000,0.0000,0.0000,0.0000",
  "depth_image": "/9j/4AAQS...",
  "gimbal_pose": "0.0052,0.0052,9.8943,0.0000,0.0000,0.0000"
}
```

### `timestamp` ###
A string containing the, [unix timestamp](https://en.wikipedia.org/wiki/Unix_time) corresponding to when the sensor frame was captured.

### `rgb_image` ###
The base64-encoded JPG image from the color camera on the quad. This is the image that the network performs inference upon.

### `pose` ###
The pose of the quad with respect to QuadSim's world frame. The fields have the following format `[x, y, z, roll, pitch, yaw]`, each distane is expressed in meters, and each angular component is expressed in radians.

### `depth_image` ###
The base64-encoded JPG image from the depth camera on the quad. The depth image is used to compute in conjunction with the color image and the pose information to compute the location of the detected object in the world frame.

### `gimbal pose ` ###
The pose of the quad's camera gimbal with respect to the quad's body frame. This pose is used in conjunction with the quad body frame pose is required to determine the coordinates of the detected object in the world frame. This field shares it's formatting with the `pose` attribute.

## `object_detected` event (emitted by Python code) ##
These events are published by your Python code each time an object of interest is detected in a sensor frame. Upon recieving an `object_detected` event, QuadSim will begin to move towards the detected object.

Example
```
{
  "coords": [0.0, 1.0, 2.0]"
}
```

### `coords` ###
Describes the coordinates of the detected objcet, in QuadSim's world frame.

## `create_box_marker` event (emitted by Python code) ##
Creates a box marker that can be used to visualize the location of objects objects in the QuadSim world.

Example
```
{
  "id": 0,
  "pose": [0.0, 1.0, 2.0, 3.14, 3.14, 1.57],
  "dimensions": [1.0,1.0,1.0],
  "color": [0,1,0,0.8],
  "duration": 1.0
}
```

### `id` ###
Identified used to reference the box after it has been created. Currently only deletion of the marker is supported.

### `pose` ###
The pose of the marker in the QuadSim world frame. The fields have the following format `[x, y, z, roll, pitch, yaw]`, each distane is expressed in meters, and each angular component is expressed in radians.

### `dimensions` ###
The dimensions of the box marker, units in meters, `[length, width, depth]`.

### `color` ###
The color of the box, including the alpha value. All values range from zero to one. An alpha value of 0.0 corresponds to complete transparency, while a value of 1.0 is completely opaque `[r, g, b, a]`

### `duration` ###
The time duration for which the box marker will exist, represented in seconds. A value of -1 indicates that the marker will exist indefinitely.


## `delete_marker` event (emitted by Python code) ##
This event is used to delete markers.

Example
```
{
  "id": 0
}
```

### `id` ###
The id of the marker to be deleted. A value of -1 will cause all markers to be deleted.
