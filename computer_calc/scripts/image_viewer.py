#! /usr/bin/python

import rospy
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge, CvBridgeError
import cv2


bridge = CvBridge()
counter = 0


def image_callback(msg):
    global counter
    try:
        cv2_img = bridge.compressed_imgmsg_to_cv2(msg)
    except CvBridgeError, e:
        print(e)
    else:
        cv2.imshow("Camera", cv2_img)
        cv2.waitKey(1)


def main():
    rospy.init_node('image_saver')
    image_topic = "/camera/image/compressed"
    rospy.Subscriber(image_topic, CompressedImage, image_callback)
    rospy.spin()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
