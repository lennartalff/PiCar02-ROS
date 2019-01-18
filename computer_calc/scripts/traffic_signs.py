#!/usr/bin/env python

import rospy

import numpy as np
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Int16

IMG_WIDTH = 640
IMG_HEIGHT = 480

FIRST_LINE_IDX = 460
SECOND_LINE_IDX = 440
THIRD_LINE_IDX = 420
FIRST_LINE_ROI = np.index_exp[FIRST_LINE_IDX:FIRST_LINE_IDX+1, :]
SECOND_LINE_ROI = np.index_exp[SECOND_LINE_IDX:SECOND_LINE_IDX+1, :]
THIRD_LINE_RIO = np.index_exp[THIRD_LINE_IDX:THIRD_LINE_IDX+1, :]
# For HSV, Hue range is [0,179], Saturation range is [0,255] and
# Value range is [0,255]
LOWER_GREEN_HSV = np.array([45, 30, 30])
UPPER_GREEN_HSV = np.array([85, 255, 255])

LOWER_RED_1_HSV = np.array([0, 40, 40])
UPPER_RED_1_HSV = np.array([7, 255, 255])
LOWER_RED_2_HSV = np.array([165, 40, 40])
UPPER_RED_2_HSV = np.array([179, 255, 255])

LOWER_BLUE_HSV = np.array([90, 30, 30])
UPPER_BLUE_HSV = np.array([130, 255, 255])

CONTOUR_AREA_THRESHOLD = 1000

SPEED_0 = 900
SPEED_1 = 1500
SPEED_2 = 2000
SPEED_3 = 3000

LEFT = 0
RIGHT = 1
STRAIGHT = 2
STOP = 3
TS_SPEED_0 = 4
TS_SPEED_1 = 5
TS_SPEED_2 = 6
TS_SPEED_3 = 7

TS_MASK_COUNT_THRESHOLD = 0.2
TS_LEFT_RIGHT_THRESHOLD = 1.4

CAMERA_HEIGHT = 272.0       # mm
SENSOR_WIDTH = 3.6736       # mm
SENSOR_HEIGHT = 2.75968     # mm
FOCAL_LENGTH_PIX = 480.0
PRINCIPAL_POINT = (312, 243)
CAMERA_ROTATION = np.mat([
    [0.9979, -0.0252, -0.0595],
    [0.0401, 0.9638, 0.2636],
    [0.0507, -0.2654, 0.9628]
])


def pix2world(pt, u0, v0, h):
    pt_screen = np.array(pt)
    x_screen = (pt_screen - np.array([float(PRINCIPAL_POINT[0] + 0.5), float(PRINCIPAL_POINT[1] + 0.5)]))
    x_screen = np.array([x_screen[0], x_screen[1], 1.0])
    x_screen[0] = x_screen[0] / FOCAL_LENGTH_PIX
    x_screen[1] = x_screen[1] / FOCAL_LENGTH_PIX
    x = CAMERA_ROTATION * np.transpose(np.mat(x_screen))
    x_world = (h / x[1]) * np.transpose(x)
    return x_world


class Rectangle(object):
    def __init__(self, x=0, y=0, w=0, h=0):
        self.x = x
        self.y = y
        self.w = w
        self.h = h


class TrafficSign(Rectangle):
    def __init__(self, x=0, y=0, w=0, h=0, ts_type=None):
        super(TrafficSign, self).__init__(x=x, y=y, w=w, h=h)
        self.ts_type = ts_type


class ImageProcessor(object):
    def __init__(self):
        # subscribe to camera images published by the PiCar
        self.subscriber = rospy.Subscriber(
            "/camera/image/compressed",
            CompressedImage,
            self.img_rcv_cb,
            queue_size=1,
            buff_size=2**20
        )
        # publish points needed by the controller
        self.pub_point = rospy.Publisher(
            "/tracking_point/PointStamped",
            PointStamped,
            queue_size=1
        )
        self.pub_throttle = rospy.Publisher(
            "/target_throttle/Int16",
            Int16,
            queue_size=1
        )
        self.bridge = CvBridge()
        self.ts = TrafficSign()
        self.lane_boundaries_screen = [[0, 0], [0, 0]]
        self.direction = LEFT
        self.speed = SPEED_0
        self.stop = False
        self.point_msg = PointStamped()
        self.throttle_msg = Int16()

        self.img_received = np.zeros([IMG_HEIGHT, IMG_WIDTH, 3], dtype=np.uint8)
        self.img_overlay = np.zeros([IMG_HEIGHT, IMG_WIDTH, 3], dtype=np.uint8)
        self.img_ts_copy = np.zeros([IMG_HEIGHT, IMG_WIDTH, 3], dtype=np.uint8)
        self.img_ts_mask = np.zeros([IMG_HEIGHT, IMG_WIDTH], dtype=np.uint8)
        self.img_lane_copy = np.zeros([IMG_HEIGHT, IMG_WIDTH, 3], dtype=np.uint8)
        self.img_lane_mask_1 = np.zeros([IMG_HEIGHT, IMG_WIDTH], dtype=np.uint8)
        self.img_lane_mask_2 = np.zeros([IMG_HEIGHT, IMG_WIDTH], dtype=np.uint8)
        self.img_lane_contour = np.zeros([IMG_HEIGHT, IMG_WIDTH], dtype=np.uint8)
        self.img_lane_LOI = np.zeros([IMG_HEIGHT, IMG_WIDTH], dtype=np.uint8)
        self.img_lane_LOI[FIRST_LINE_ROI] = 255
        self.img_lane_LOI[SECOND_LINE_ROI] = 255
        self.img_lane_LOI[THIRD_LINE_RIO] = 255

    def img_rcv_cb(self, msg):
        # store and copy received image
        self.img_received = self.bridge.compressed_imgmsg_to_cv2(msg)
        self.img_overlay[:] = self.img_received[:]
        cv2.cvtColor(self.img_received, cv2.COLOR_BGR2HSV, self.img_received)
        self.img_lane_copy[:] = self.img_received[:]
        self.img_ts_copy[:] = self.img_received[:]
        cv2.imshow("original", self.img_received)
        cv2.waitKey(1)

        # LANE DETECTION
        cv2.inRange(self.img_ts_copy, LOWER_GREEN_HSV, UPPER_GREEN_HSV, self.img_ts_mask)
        _, contours, _ = cv2.findContours(self.img_ts_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(contour) > 300:
                self.ts.x, self.ts.y, self.ts.w, self.ts.h = cv2.boundingRect(contour)
                cv2.rectangle(self.img_overlay,
                              (self.ts.x, self.ts.y),
                              (self.ts.x + self.ts.w,
                               self.ts.y + self.ts.h),
                              (0, 255, 0),
                              2)
                self.img_lane_copy[self.ts.y:self.ts.y+self.ts.h, self.ts.x:self.ts.x+self.ts.w, :] = 0
                if self.ts.h > 50 and self.ts.w > 50:
                    self.ts_detection()
                    self.react_to_ts()

        cv2.imshow("traffic_sign_mask", self.img_ts_mask)
        cv2.waitKey(1)
        # create mask
        cv2.inRange(self.img_lane_copy, LOWER_RED_1_HSV, UPPER_RED_1_HSV, self.img_lane_mask_1)
        cv2.inRange(self.img_lane_copy, LOWER_RED_2_HSV, UPPER_RED_2_HSV, self.img_lane_mask_2)
        self.img_lane_mask_1 = self.img_lane_mask_1 | self.img_lane_mask_2
        cv2.imshow("lane_mask", self.img_lane_mask_1)
        cv2.waitKey(1)
        cv2.erode(self.img_lane_mask_1, np.ones([5, 5], np.uint8), self.img_lane_mask_1)
        cv2.dilate(self.img_lane_mask_1, np.ones([5, 5], np.uint8), self.img_lane_mask_1)
        _, contours, _ = cv2.findContours(self.img_lane_mask_1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # default case: not moving -> only changed if lane is found and car is not stopping
        (self.point_msg.point.x, self.point_msg.point.y, self.point_msg.point.z) = (0, 0, 0)
        self.point_msg.header.stamp = rospy.Time.now()

        # at least one contour has to exist, otherwise no lane is visible
        if len(contours) > 0:
            # sort contours in descending order regarding the contours' area
            contours.sort(key=cv2.contourArea, reverse=True)

            # check if it's likely, that the biggest contour refers to a lane
            if cv2.contourArea(contours[0]) > CONTOUR_AREA_THRESHOLD:
                cv2.drawContours(self.img_overlay, contours, 0, (255, 0, 0), 2)
                self.img_lane_contour = cv2.drawContours(np.zeros([IMG_HEIGHT, IMG_WIDTH], dtype=np.uint8), contours, 0, 255, 1)

                # check if second contour exists and if it's likely that the second biggest contour also refers to a
                # lane
                if len(contours) > 1 and cv2.contourArea(contours[1]) > CONTOUR_AREA_THRESHOLD:
                    cv2.drawContours(self.img_overlay, contours, 1, (255, 0, 0), 2)
                    cv2.drawContours(self.img_lane_contour, contours, 1, 255, 1)

                # find the intersections of the lane contour(s) and the line of interest
                index = np.nonzero(self.img_lane_contour[SECOND_LINE_ROI])
                if len(index) > 0 and len(index[0] > 0):
                    cv2.circle(self.img_overlay, (index[1][0], SECOND_LINE_IDX), 10, (255, 255, 0), 2)
                    cv2.circle(self.img_overlay, (index[1][-1], SECOND_LINE_IDX), 10, (255, 255, 0), 2)
                    self.lane_boundaries_screen[0] = (index[1][0], SECOND_LINE_IDX)
                    self.lane_boundaries_screen[1] = (index[1][-1], SECOND_LINE_IDX)
                    if not self.stop:
                        if self.direction == LEFT:
                            point_3d = pix2world(self.lane_boundaries_screen[0], IMG_WIDTH, IMG_HEIGHT, CAMERA_HEIGHT)
                        else:
                            point_3d = pix2world(self.lane_boundaries_screen[1], IMG_WIDTH, IMG_HEIGHT, CAMERA_HEIGHT)

                        (self.point_msg.point.x, self.point_msg.point.y, self.point_msg.point.z) = point_3d.tolist()[0]

        # publish the point message for the controller
        self.pub_point.publish(self.point_msg)
        self.throttle_msg.data = self.speed
        self.pub_throttle.publish(self.throttle_msg)

        # draw some stuff
        cv2.imshow("lane_mask_refined", self.img_lane_mask_1)
        cv2.waitKey(1)

        cv2.imshow("Overlay", self.img_overlay)
        cv2.waitKey(1)

    def ts_detection(self):
        img = self.img_ts_copy[self.ts.y:self.ts.y+self.ts.h, self.ts.x:self.ts.x+self.ts.w]
        mask_b = cv2.inRange(img, LOWER_BLUE_HSV, UPPER_BLUE_HSV)
        mask_r_1 = cv2.inRange(img, LOWER_RED_1_HSV, UPPER_RED_1_HSV)
        mask_r_2 = cv2.inRange(img, LOWER_RED_2_HSV, UPPER_RED_2_HSV)
        mask_r_1 = mask_r_1 | mask_r_2
        cv2.imshow("ts_red_mask", mask_r_1)
        cv2.waitKey(1)

        if 0 and np.count_nonzero(mask_b) > mask_b.size*TS_MASK_COUNT_THRESHOLD:
            print("Blue ts found")
            stop = False
            for col in range(self.ts.w):
                index_left = np.nonzero(mask_b[:, col])
                if len(index_left[0]) > 0:
                    stop = True
                    index_left = (index_left[0][0], col)
                if stop:
                    break
            stop = False
            for col_rev in range(self.ts.w):
                index_right = np.nonzero(mask_b[:, -1-col_rev])
                if len(index_right[0] > 0):
                    stop = True
                    index_right = (index_right[0][0], col)
                if stop:
                    break
            pixel_count_l = np.count_nonzero(mask_b[:, index_left[1]+2])
            pixel_count_r = np.count_nonzero(mask_b[:, index_right[1]-2])
            if pixel_count_l > pixel_count_r*TS_LEFT_RIGHT_THRESHOLD:
                self.ts.ts_type = TS_SPEED_3
            elif pixel_count_r > pixel_count_l*TS_LEFT_RIGHT_THRESHOLD:
                self.ts.ts_type = TS_SPEED_1
            else:
                # TODO: Dreieck, das nach oben oder unten zeigt
                self.ts.ts_type = None
                
        elif np.count_nonzero(mask_r_1) > mask_r_1.size*TS_MASK_COUNT_THRESHOLD:
            print("red ts found")
            stop = False
            for col in range(self.ts.w):
                index_left = np.nonzero(mask_r_1[:, col])
                if len(index_left[0]) > 0:
                    stop = True
                    index_left = (index_left[0][0], col)
                if stop:
                    break
            stop = False
            for col_rev in range(self.ts.w):
                index_right = np.nonzero(mask_r_1[:, self.ts.w-1-col_rev])
                if len(index_right[0] > 0):
                    stop = True
                    index_right = (index_right[0][0], self.ts.w-1-col_rev)
                if stop:
                    break

            print("Index_l = {}, Index_r= {}".format(index_left, index_right))
            pixel_count_l = np.count_nonzero(mask_r_1[:, index_left[1]+2])
            pixel_count_r = np.count_nonzero(mask_r_1[:, index_right[1]-2])
            print("left:{}, right:{}".format(pixel_count_l, pixel_count_r))
            if float(pixel_count_l) > float(pixel_count_r*TS_LEFT_RIGHT_THRESHOLD):
                self.ts.ts_type = RIGHT
                print("RIGHT")
            elif pixel_count_r > pixel_count_l*TS_LEFT_RIGHT_THRESHOLD:
                self.ts.ts_type = LEFT
                print("LEFT")
            else:
                # TODO: Dreieck, das nach oben oder unten zeigt
                self.ts.ts_type = None
                print("ups")

        else:
            self.ts.ts_type = None
                
    def react_to_ts(self):
        if self.ts.ts_type is None:
            return
        elif self.ts.ts_type is LEFT:
            self.direction = LEFT
            print("direction = left")
        elif self.ts.ts_type is RIGHT:
            self.direction = RIGHT
            print("direction = right")
        # TODO: All other traffic signs


def main():
    rospy.init_node("traffic_signs_node")
    ImageProcessor()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down traffic_signs_node")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
