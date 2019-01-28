#!/usr/bin/env python
import rospy
from picamera import PiCamera
import io
import time
from sensor_msgs.msg import CompressedImage


class FrameSplitter(object):
    def __init__(self):
        self.done = False
        self.publisher = rospy.Publisher(
            "/camera/image/compressed",
            CompressedImage,
            queue_size=1
        )
        self.msg = CompressedImage()
        self.stream = io.BytesIO()

    def write(self, buf):
        # JPEGs start with the magic number FF D8, so it is used to identify
        # the beginning of a new frame
        if buf.startswith(b'\xff\xd8'):
            size = self.stream.tell()
            if size > 0:
                self.stream.seek(0)
                self.msg.data = self.stream.read(size)
                self.msg.format = "jpeg"
                self.msg.header.stamp = rospy.Time.now()
                try:
                    self.publisher.publish(self.msg)
                except rospy.ROSException:
                    pass
                self.stream.seek(0)
                self.stream.truncate()
        self.stream.write(buf)


def main():
    rospy.init_node('camera_node', anonymous=True)
    camera_settings = rospy.get_param("camera_settings")
    with PiCamera(resolution=(camera_settings["width"], camera_settings["height"]), framerate=camera_settings["framerate"]) as camera:
        camera.iso = 800
        camera.vflip = True
        camera.hflip = True
        time.sleep(2)
        camera.shutter_speed = camera.exposure_speed
        camera.exposure_mode = 'off'
        g = camera.awb_gains
        camera.awb_mode = 'off'
        camera.awb_gains = g
        if camera_settings["racemode"] == 1:
            camera.zoom = (0.0, 0.0, 1.0, 1.0)
        output = FrameSplitter()
        camera.start_recording(output, format='mjpeg')
        while not rospy.is_shutdown():
            try:
                camera.wait_recording(1)
            except KeyboardInterrupt:
                print("Shutting down by KeyboardInterrupt")
                break
        camera.stop_recording()
        print("Shutting down camera node")


if __name__ == '__main__':
    main()
