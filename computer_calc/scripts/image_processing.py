#!/usr/bin/env python

import sys, time

import cv2
import rospy
import numpy as np
from cv_bridge import CvBridge
import imutils

from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import PointStamped

CAMERA_HEIGHT = 272.0     # mm
SENSOR_WIDTH = 3.6736   # mm
SENSOR_HEIGHT = 2.75968    # mm
FOCAL_LENGTH_PIX = 314.0
CAMERA_ROTATION = np.mat([
    [0.9979, -0.0252, -0.0595],
    [0.0401, 0.9638, 0.2636],
    [0.0507, -0.2654, 0.9628]
])

class ImageProcessor(object):
    def __init__(self):
        self.subscriber = rospy.Subscriber(
            "/raspicam_node/image/compressed",
            CompressedImage,
            self.img_rcv_callback,
            queue_size=1,
            buff_size=2**20)
        self.publisher = rospy.Publisher(
            "/tracking_point/PointStamped",
            PointStamped,
            queue_size=1
        )
        self.br = CvBridge()
        self.init_blob_detector()

    def init_blob_detector(self):
        params = cv2.SimpleBlobDetector_Params()

        params.minThreshold = 0
        params.maxThreshold = 255

        params.filterByArea = True
        params.minArea = 10

        params.filterByCircularity = False
        params.filterByConvexity = False
        params.filterByInertia = False


        self.blobDetector = cv2.SimpleBlobDetector_create(params)


    def img_rcv_callback(self, msg):
        img = self.br.compressed_imgmsg_to_cv2(msg)
        cv2.imshow('cv_img', img)
        cv2.waitKey(1)
        mask = self.color_detection(img)
        mask = self.refine_mask(mask)
        point_pixel = self.find_contours(mask)
        if not point_pixel == None:
            point_3d = self.pix2world(point_pixel, 360, 240, CAMERA_HEIGHT-155)
            msg = PointStamped()
            msg.header.stamp = rospy.Time.now()
            (msg.point.x, msg.point.y, msg.point.z) = point_3d.tolist()[0]
            self.publisher.publish(msg)
        #self.detect_blobs(mask)

    def color_detection(self, img_bgr,
        lower_color_hsv=np.array([45, 30, 30]),
        upper_color_hsv=np.array([85, 255, 255]) ):
        # For HSV, Hue range is [0,179], Saturation range is [0,255] and
        # Value range is [0,255]
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(img_hsv, lower_color_hsv, upper_color_hsv)
        return mask

    def refine_mask(self, mask):
        kernel = np.ones((5,5), np.uint8)
        #mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        cv2.imshow('cv_mask refined', mask)
        cv2.waitKey(1)
        return mask

    def find_contours(self, mask):
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        center = None

        if len(cnts) > 0:
    		# find the largest contour in the mask, then use
    		# it to compute the minimum enclosing circle and
    		# centroid
    		c = max(cnts, key=cv2.contourArea)
    		((x, y), radius) = cv2.minEnclosingCircle(c)
    		M = cv2.moments(c)
    		center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        return center

    def pix2world(self, pt, u0, v0, h):
        pixel_width = SENSOR_WIDTH/u0
        pixel_height = SENSOR_HEIGHT/v0
        pt_screen = np.array(pt)
        x_screen = (pt_screen - np.array([float(u0/2+0.5), float(v0/2+0.5)]))
        x_screen = np.array([x_screen[0], x_screen[1], 1.0])
        x_screen[0] = x_screen[0] / FOCAL_LENGTH_PIX
        x_screen[1] = x_screen[1] / FOCAL_LENGTH_PIX
        x = CAMERA_ROTATION * np.transpose(np.mat(x_screen))
        x_world = (h/x[1]) * np.transpose(x)
        return(x_world)

    def detect_blobs(self, img):
        img = 255-img
        keyPoints = self.blobDetector.detect(img)
        img_blob = cv2.drawKeypoints(img, keyPoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow('Blobs', img_blob)
        pts = [p.pt for p in keyPoints]
        print(pts)
        cv2.waitKey(1)


def main(args):
    img_processor = ImageProcessor()
    rospy.init_node('image_processing_node', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down image_processing_node")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
