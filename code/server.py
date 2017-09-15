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

import eventlet.wsgi
import socketio
from flask import Flask

# Create socketio server and Flask app
sio = socketio.Server()
app = Flask(__name__)


@sio.on('create_box_marker')
def create_box_marker(sid, data):
    print(data)
    sio.emit('create_box_marker', data, skip_sid=sid)


@sio.on('object_lost')
def target_lost(sid, data):
    print(data, 'object_lost')
    sio.emit('object_lost', data, skip_sid=sid)


@sio.on('delete_marker')
def delete_marker(sid, data):
    sio.emit('delete_marker', data, skip_sid=sid)


@sio.on('object_detected')
def object_detected(sid, data):
    print(data)
    sio.emit('object_detected', data, skip_sid=sid)


@sio.on('sensor_frame')
def sensor_frame(sid, data):
    sio.emit('sensor_data', data, skip_sid=sid)


if __name__ == '__main__':
    # wrap Flask application with socketio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
