#!/usr/bin/env python
import rospy
from picamera import PiCamera
import io
import time
from sensor_msgs.msg import CompressedImage


class FrameSplitter(object):
    def __init__(self):
        self.done = False
        # Construct a pool of 4 image processors along with a lock
        # to control access between threads
        self.publisher = rospy.Publisher(
            "/camera/image/compressed",
            CompressedImage,
            queue_size=1
        )
        self.msg = CompressedImage()
        self.stream = io.BytesIO()

    def write(self, buf):
        if buf.startswith(b'\xff\xd8'):
            size = self.stream.tell()
            if size > 0:
                self.stream.seek(0)
                self.msg.data = self.stream.read(size)
                self.msg.format = "jpeg"
                self.msg.header.stamp = rospy.Time.now()
                self.publisher.publish(self.msg)
                self.stream.seek(0)
                self.stream.truncate()
        self.stream.write(buf)


def main():
    rospy.init_node('camera_node', anonymous=True)
    with PiCamera(resolution=(640, 480), framerate=40) as camera:
        camera.iso = 800
        time.sleep(2)
        camera.shutter_speed = camera.exposure_speed
        camera.exposure_mode = 'off'
        g = camera.awb_gains
        camera.awb_mode = 'off'
        camera.awb_gains = g
        output = FrameSplitter()
        camera.start_recording(output, format='mjpeg')
        while not rospy.is_shutdown():
            try:
                camera.wait_recording(1)
            except KeyboardInterrupt:
                print("Shuttding down by KeyboardInterrupt")
                break
        camera.stop_recording()
        print("Shuttding down camera node")


if __name__ == '__main__':
    main()
