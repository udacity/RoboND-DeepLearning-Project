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

from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
import pyqtgraph as pg
import pyqtgraph.ptime as ptime
from scipy import misc
import glob
import os
import time
from multiprocessing import Process, Manager, Queue
import sched, time, threading
from socketIO_client import SocketIO, LoggingNamespace
from PIL import Image
from io import BytesIO
from scipy import misc
import base64
import argparse
from utils import data_iterator

# https://gist.githubusercontent.com/Overdrivr/1c505f9a4ec5be35b2a1/raw/297ca3b986f974d93f06921b0c3baabe021562be/Plot.py
# very useful gist:)
# This function is responsible for displaying the data
# it is run in its own process to liberate main process
class SideBySidePlot():
    def __init__(self,name, image_hw):
        self.name = name
        self.image_hw = image_hw
        # Process-local buffers used to host the displayed data

    def start(self):
        self.q = Queue()
        self.p = Process(target=self.run)
        self.p.start()
        return self.q

    def join(self):
        self.p.join()

    def _update(self):
        while not self.q.empty():
            item = self.q.get()
            label = (item[1]*255).astype(np.int)
            image = (item[0]*255).astype(np.int)
            
            plot_image = np.concatenate((image, label), 1)
            plot_image = np.rot90(plot_image, -1)
            self.img.setImage(plot_image)

    def run(self):
        app = QtGui.QApplication([])
        ## Create window with GraphicsView widget
        win = pg.GraphicsLayoutWidget()
        win.show()  ## show widget alone in its own window
        win.setWindowTitle('pyqtgraph example: ImageItem')
        view = win.addViewBox()

        ## lock the aspect ratio so pixels are always square
        view.setAspectLocked(True)

        ## Create image item
        self.img = pg.ImageItem(border='w')
        view.addItem(self.img)

        ## Set initial view bounds
        view.setRange(QtCore.QRectF(0, 0, 2*self.image_hw, self.image_hw))

        timer = QtCore.QTimer()
        timer.timeout.connect(self._update)
        timer.start(50)

        app.exec_()

class OverlayPlot():
    def __init__(self,name,image_hw):
        self.name = name
        self.image_hw = image_hw
        # Process-local buffers used to host the displayed data

    def start(self):
        self.q = Queue()
        self.p = Process(target=self.run)
        self.p.start()
        return self.q

    def join(self):
        self.p.join()

    def _update(self):
        while not self.q.empty():
            item = self.q.get()
            image = item[0]
            pred = item[1]
            result = overlay_predictions(image, pred, None, 0.5, 1)
            result = result.convert('RGB')
            result = np.asarray(result)

            result = np.rot90(result, -1)
            self.img.setImage(result)

    def run(self):
        app = QtGui.QApplication([])
        ## Create window with GraphicsView widget
        win = pg.GraphicsLayoutWidget()
        win.show()  ## show widget alone in its own window
        win.setWindowTitle('pyqtgraph example: ImageItem')
        view = win.addViewBox()

        ## lock the aspect ratio so pixels are always square
        view.setAspectLocked(True)

        ## Create image item
        self.img = pg.ImageItem(border='w')
        view.addItem(self.img)

        ## Set initial view bounds
        view.setRange(QtCore.QRectF(0, 0, self.image_hw, self.image_hw))

        timer = QtCore.QTimer()
        timer.timeout.connect(self._update)
        timer.start(50)

        app.exec_()


def overlay_predictions(image, im_softmax,image_shape, threshold, channel, seg_color=(0,255,0,172)):
    """creates a overlay using pixels with p(class) > threshold"""

    segmentation = np.expand_dims(im_softmax[:,:,channel] > threshold, 2)
    mask = segmentation * np.reshape(np.array(seg_color), (1,1,-1))
    mask = misc.toimage(mask, mode="RGBA")
    street_im = misc.toimage(image)
    street_im.paste(mask, box=None, mask=mask)
    return street_im
