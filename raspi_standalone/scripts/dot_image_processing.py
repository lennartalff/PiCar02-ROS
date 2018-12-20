#!/usr/bin/env python
import rospy
from picamera import PiCamera
from picamera.array import PiRGBArray
import io
import time
import threading
import numpy as np
import cv2
from sensor_msgs.msg import CompressedImage


class VideoCapture():
    def __init__(self, owner, resolution=(320,240), framerate=30):
        self.owner = owner
        self.camera = PiCamera()
        self.camera.resolution = resolution
        self.camera.framerate = framerate
        self.rawCapture = PiRGBArray(self.camera, size=resolution)
        self.stream = self.camera.capture_continuous(self.rawCapture,
                                                     format="bgr",
                                                     use_video_port=True)
        self.camera.iso = 800
        time.sleep(2)
        self.camera.shutter_speed = self.camera.exposure_speed
        self.camera.exposure_mode = 'off'
        g = self.camera.awb_gains
        self.camera.awb_mode = 'off'
        self.camera.awb_gains = g
        self.frame = None
        self.stopped = False

    def start(self):
        rospy.loginfo("Starting VideoCapture")
        threading.Thread(target=self.capture, args=()).start()

    def capture(self):
        for f in self.stream:
            with self.owner.image_lock:
                self.frame = f.array
                self.owner.frame_event.set()
            self.rawCapture.truncate(0)
            if self.stopped:
                self.stream.close()
                self.rawCapture.close()
                self.camera.close()
                return

    def read(self):
        with self.owner.image_lock:
            return self.frame

    def stop(self):
        self.stopped = True


class ImageProcessor():
    def __init__(self):
        self.image_lock = threading.Lock()
        self.frame_event = threading.Event()
        self.frame_event.clear()
        self.cap = VideoCapture(self)
        self.image = None

    def start(self):
        self.cap.start()
        self.lower_color_hsv = np.array([45, 30, 30])
        self.upper_color_hsv = np.array([85, 255, 255])
        while not rospy.is_shutdown():
            self.frame_event.wait()
            self.process_frame()
            self.frame_event.clear()

    def process_frame(self):
        rospy.loginfo("Reading frame")
        frame = self.cap.read()

    def stop(self):
        if self.cap:
            self.cap.stop()


def main():
    rospy.init_node('camera_node')
    vc = ImageProcessor()
    rospy.on_shutdown(vc.stop)
    rospy.loginfo("Created ImageProcessor")
    vc.start()
    print("Shutting down camera node")


if __name__ == '__main__':
    main()
