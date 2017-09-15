#!/usr/bin/env python

import time
from socketIO_client import SocketIO, LoggingNamespace

class TestTargetLost(object):
    def __init__(self, wait_time=1):
        self.wait_time = wait_time
        self.last_mode_change = 0
        self.mode = 'target_lost'


    def on_sensor_data(self, data):
        timestamp = float(data['timestamp'])
        dt = timestamp - self.last_mode_change

        if dt > self.wait_time and self.mode=='object_detected':
            self.last_mode_change = timestamp
            self.mode = 'target_lost'
            self.last_mode_change = 0
            socketIO.emit(self.mode, '')
          
        if dt > self.wait_time and self.mode=='target_lost':
            self.last_mode_change = timestamp
            self.mode = 'object_detected'
            self.last_mode_change = 0
            socketIO.emit(self.mode, {'coords':[0,0,10]})

def on_disconnect():
    print('disconnect')


def on_connect():
    print('connect')


def on_reconnect():
    print('reconnect')


if __name__ == '__main__':
    tester = TestTargetLost()
    socketIO = SocketIO('localhost', 4567, LoggingNamespace)
    socketIO.on('connect', on_connect)
    socketIO.on('disconnect', on_disconnect)
    socketIO.on('reconnect', on_reconnect)
    socketIO.on('sensor_data', tester.on_sensor_data)

    for i in range(100):
        tester.on_sensor_data({"timestamp":str(time.time()+i*3)})
        time.sleep(0.2)
    
    #socketIO.wait(seconds=100000000)
