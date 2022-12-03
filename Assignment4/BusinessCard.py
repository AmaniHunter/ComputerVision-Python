from imutils.video import FPS
from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import imutils
import time
import cv2
import pytesseract
import webbrowser
import re
import depthai as dai

pipeline = dai.Pipeline()

camRgb = pipeline.create(dai.node.ColorCamera)
xoutVideo = pipeline.create(dai.node.XLinkOut)

xoutVideo.setStreamName("face detector")

camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setVideoSize(860, 720)

xoutVideo.input.setBlocking(False)
xoutVideo.input.setQueueSize(1)

camRgb.video.link(xoutVideo.input)
#vid = cv2.VideoCapture(0)
with dai.Device(pipeline) as device:
    video = device.getOutputQueue(name="face detector", maxSize=1, blocking=False)

    while True:
        videoIn = video.get()
        frame = videoIn.getCvFrame()

        cv2.imshow('Face Detector', frame)
        if cv2.waitKey(1) == ord('q'):
            break
cv2.destroyAllWindows()
