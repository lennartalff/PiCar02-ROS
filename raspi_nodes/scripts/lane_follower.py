#!/usr/bin/env python

import rospy

import numpy as np
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage
from raspi_nodes.msg import LaneFollowerMsg
from geometry_msgs.msg import Point
from std_msgs.msg import Float32, Int16
import time
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
import tensorflow as tf


# For HSV, Hue range is [0,179], Saturation range is [0,255] and
# Value range is [0,255]
LOWER_GREEN_HSV = np.array([45, 30, 30])
UPPER_GREEN_HSV = np.array([85, 255, 255])

LOWER_RED_1_HSV = np.array([0, 60, 40])
UPPER_RED_1_HSV = np.array([7, 255, 255])
LOWER_RED_2_HSV = np.array([165, 60, 40])
UPPER_RED_2_HSV = np.array([179, 255, 255])

LOWER_BLUE_HSV = np.array([90, 90, 40])
UPPER_BLUE_HSV = np.array([130, 255, 255])


DIR_LEFT = 0
DIR_RIGHT = 1
DIR_STRAIGHT = 2

SPEED_0 = 800
SPEED_1 = 1200
SPEED_2 = 1500
SPEED_3 = 2000

TS_LEFT = 0
TS_RIGHT = 1
TS_STRAIGHT = 2
TS_STOP = 3
TS_SPEED_0 = 4
TS_SPEED_1 = 5
TS_SPEED_2 = 6
TS_SPEED_3 = 7


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
        self.ts_type_last = None
        self.last_time = 0


class LaneFollower(object):
    def __init__(self, width, height, nn):
        self.nn = nn
        self.nn_model = load_trained_model("/home/lennartalff/catkin_ws/src/computer_calc/scripts/2500_initial_softmax.h5")
        self.train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)
        self.train_gen = self.train_datagen.flow_from_directory(
            "/home/lennartalff/catkin_ws/src/computer_calc/pictures_train",
            target_size=(60, 60),
            batch_size=50,
            class_mode='categorical')
        self.graph = tf.get_default_graph()
        self.IMG_WIDTH = width
        self.IMG_HEIGHT = height
        self.SLICE_HEIGHT = self.IMG_HEIGHT/12
        self.lower_slice_row_idx = self.IMG_HEIGHT*9/12
        self.lower_slice = np.index_exp[self.lower_slice_row_idx-self.SLICE_HEIGHT/2:self.lower_slice_row_idx+self.SLICE_HEIGHT/2, :]
        self.upper_slice_row_idx = self.IMG_HEIGHT*7/12
        self.upper_slice = np.index_exp[self.upper_slice_row_idx-self.SLICE_HEIGHT/2:self.upper_slice_row_idx+self.SLICE_HEIGHT/2, :]

        self.sub_img = rospy.Subscriber(
            "/camera/image/compressed",
            CompressedImage,
            self.img_rcv_cb,
            queue_size=1,
            buff_size=2**20
        )
        self.pub_output = rospy.Publisher(
            "/lane_follower/output",
            LaneFollowerMsg,
            queue_size=1
        )
        self.pub_visualizer = rospy.Publisher(
            "/visualizer/camera_image",
            CompressedImage,
            queue_size=1
        )
        self.pub_normalized_control_error = rospy.Publisher(
            "/lane_follower/normalized_control_error",
            Float32,
            queue_size=1
        )
        self.pub_control_speed = rospy.Publisher(
            "/lane_follower/target_speed",
            Int16,
            queue_size=1
        )
        self.br = CvBridge()
        self.img = np.zeros([self.IMG_HEIGHT, self.IMG_WIDTH, 3], dtype=np.uint8)

        self.contour_area_threshold_track = 1000 * width / 640

        # used to save traffic sign information
        self.ts = TrafficSign()
        # flag that indicates whether a stop sign was registered and the car should stop at the next line
        self.stop_registered = False
        self.crossing_detected_old = False
        self.crossing_detected = False
        self.crossing_detected_debounce = 0x00
        self.nrDebounce = 0x03

        # status variable that indicates the movement direction
        self.direction = DIR_RIGHT
        # status variable that sets the car's speed
        self.speed = SPEED_0

    def img_rcv_cb(self, msg):
        self.ts.ts_type = None

        # republish untouched camera image for the visualizer
        self.republish_image(msg)

        # decode camera jpeg to opencv image
        self.img = self.br.compressed_imgmsg_to_cv2(msg)

        # convert to HSV colors and create slices
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        img_lower_slice = self.img[self.lower_slice]
        img_upper_slice = self.img[self.upper_slice]

        self.handle_ts(self.img)

        # color mask
        mask_lower_slice = cv2.inRange(img_lower_slice, LOWER_RED_1_HSV, UPPER_RED_1_HSV) | cv2.inRange(img_lower_slice, LOWER_RED_2_HSV, UPPER_RED_2_HSV)
        mask_upper_slice = cv2.inRange(img_upper_slice, LOWER_RED_1_HSV, UPPER_RED_1_HSV) | cv2.inRange(img_upper_slice, LOWER_RED_2_HSV, UPPER_RED_2_HSV)

        # refine the mask to reduce noise
        cv2.erode(mask_lower_slice, np.ones([5, 5], dtype=np.uint8), mask_lower_slice)
        cv2.erode(mask_upper_slice, np.ones([5, 5], dtype=np.uint8), mask_upper_slice)
        cv2.dilate(mask_lower_slice, np.ones([5, 5], dtype=np.uint8), mask_lower_slice)
        cv2.dilate(mask_upper_slice, np.ones([5, 5], dtype=np.uint8), mask_upper_slice)

        lower_lane_boundaries = self.get_lane_boundaries(mask_lower_slice)
        upper_lane_boundaries = self.get_lane_boundaries(mask_upper_slice)

        self.crossing_detected_debounce = int(len(lower_lane_boundaries) < len(upper_lane_boundaries)) | ((self.crossing_detected_debounce << 1) & self.nrDebounce)
        if self.crossing_detected_debounce == self.nrDebounce:
            self.crossing_detected = True
        elif self.crossing_detected_debounce == 0x00:
            self.crossing_detected = False

        if len(lower_lane_boundaries) > 0:
            target_point = self.get_target_point(lower_lane_boundaries, upper_lane_boundaries)
            cv2.circle(self.img, target_point, 10, (255, 0, 0), 2)
        else:
            target_point = (0, 0)

        self.publish_lane_follower_msg(target_point, lower_lane_boundaries, upper_lane_boundaries)

        msg = Float32()
        if not target_point[1] == 0:
            msg.data = float(float((target_point[0]-self.IMG_WIDTH/2))/float((self.IMG_WIDTH/2)))
        else:
            msg.data = float(100)
        try:
            self.pub_normalized_control_error.publish(msg)
        except rospy.ROSException:
            pass
        msg = Int16()
        if self.stop_registered and self.detect_hold(img_lower_slice):
            msg.data = 0
            try:
                self.pub_control_speed.publish(msg)
            except rospy.ROSException:
                pass
            time.sleep(5)
            self.stop_registered = False
        msg.data = self.speed
        try:
            self.pub_control_speed.publish(msg)
        except rospy.ROSException:
            pass

    def republish_image(self, msg):
        m = CompressedImage()
        m.data = msg.data
        m.format = "jpeg"
        m.header.stamp = rospy.Time.now()
        try:
            self.pub_visualizer.publish(m)
        except rospy.ROSException:
            rospy.logwarn("Could not republish camera image!")

    def publish_lane_follower_msg(self, target_point, lower_lane_boundaries, upper_lane_boundaries):
        msg = LaneFollowerMsg()
        (msg.target_point_x, msg.target_point_y) = target_point
        msg.lower_slice_row = self.lower_slice_row_idx
        msg.upper_slice_row = self.upper_slice_row_idx
        msg.slice_height = self.SLICE_HEIGHT
        msg.lower_lane_boundaries = [i for sublist in lower_lane_boundaries for i in sublist]
        msg.upper_lane_boundaries = [i for sublist in upper_lane_boundaries for i in sublist]
        msg.crossing_detected = self.crossing_detected
        (msg.ts_h, msg.ts_w, msg.ts_x, msg.ts_y) = (self.ts.h, self.ts.w, self.ts.x, self.ts.y)
        if self.ts.ts_type is not None:
            msg.ts_type = self.ts.ts_type
        else:
            msg.ts_type = 100
        msg.stop_registered = int(self.stop_registered)
        msg.speed = self.speed
        msg.direction = self.direction
        try:
            self.pub_output.publish(msg)
        except rospy.ROSException:
            rospy.logwarn("Could not publish LaneFollowerMsg")

    def detect_hold(self, img):
        mask = cv2.inRange(img, LOWER_BLUE_HSV, UPPER_BLUE_HSV)
        _, contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return False
        contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(img, [contour], 0, (255, 255, 255))
        if cv2.contourArea(contour) < self.contour_area_threshold_track:
            return False
        print("hold detected")
        return True

    def get_lane_boundaries(self, mask):
        """
        Determines the column indices of the lane boundaries.
        :param mask: Color mask of the lane image.
        :return: Returns list of the column indices of the lane boundaries in the following format:
        [[left_0, right_0], ..., [left_n, right_n]]
        Left and right indices might have the same value.
        """
        (rows, cols) = mask.shape
        # find contours
        _, contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # is there at least one contour?
        if not (len(contours) > 0):
            return []

        contour_images = []

        # sort contours in descending order regarding the contour's area
        contours.sort(key=cv2.contourArea, reverse=True)

        # check if the contour is big enough to be a lane
        for i in range(len(contours)):
            if cv2.contourArea(contours[i]) < self.contour_area_threshold_track:
                break
            contour_images.append(cv2.drawContours(np.zeros(mask.shape, dtype=np.uint8), contours, i, 255, 1))
        col_indices = []
        for item in contour_images:
            all_indices = np.transpose(np.nonzero(item[rows/2]))
            if not (len(all_indices) > 0):
                continue
            # only store the left and right most column index
            col_indices.append([all_indices[0][0], all_indices[-1][0]])
        col_indices.sort(key=lambda k: k[0])
        return col_indices

    def get_target_point(self, lower_lane_boundaries, upper_lane_boundaries):
        lower_lane_center = [(i[0]+i[1])/2 for i in lower_lane_boundaries]
        upper_lane_center = [(i[0]+i[1])/2 for i in upper_lane_boundaries]
        if self.crossing_detected:
            if self.direction == DIR_LEFT:
                target_point = (upper_lane_center[0], self.upper_slice_row_idx)
            elif self.direction == DIR_RIGHT:
                target_point = (upper_lane_center[-1], self.upper_slice_row_idx)
            else:
                target_point = (upper_lane_center[len(upper_lane_center)/2], self.upper_slice_row_idx)

        else:
            if len(lower_lane_center) < 3:
                if self.direction == DIR_LEFT:
                    target_point = (lower_lane_center[0], self.lower_slice_row_idx)
                elif self.direction == DIR_RIGHT:
                    target_point = (lower_lane_center[-1], self.lower_slice_row_idx)
                else:
                    target_point = (min(lower_lane_center, key=lambda x: abs(x - self.IMG_WIDTH / 2)), self.lower_slice_row_idx)

            else:
                target_point = (min(lower_lane_center, key=lambda x: abs(x-self.IMG_WIDTH/2)), self.lower_slice_row_idx)
        return target_point

    def handle_ts(self, img):
        mask = cv2.inRange(img, LOWER_GREEN_HSV, UPPER_GREEN_HSV)
        _, contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        (self.ts.y, self.ts.x, self.ts.w, self.ts.h) = (0, 0, 0, 0)
        if len(contours) > 0:
            contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(contour) > (self.IMG_WIDTH*self.IMG_HEIGHT/256):
                self.ts.x, self.ts.y, self.ts.w, self.ts.h = cv2.boundingRect(contour)
                if abs(1-float(self.ts.h)/self.ts.w) < 0.3:
                    if self.nn:
                        self.identify_nn(img[self.ts.y:self.ts.y+self.ts.h, self.ts.x:self.ts.x+self.ts.w])
                    else:
                        self.identify_ts(img[self.ts.y:self.ts.y+self.ts.h, self.ts.x:self.ts.x+self.ts.w])
                else:
                    (self.ts.y, self.ts.x, self.ts.w, self.ts.h) = (0, 0, 0, 0)

                img[self.ts.y:self.ts.y+self.ts.h, self.ts.x:self.ts.x+self.ts.w] = 0
        if self.ts.ts_type is not None:
            if self.ts.ts_type == self.ts.ts_type_last:
                if self.ts.ts_type == TS_LEFT:
                    self.direction = DIR_LEFT
                elif self.ts.ts_type == TS_RIGHT:
                    self.direction = DIR_RIGHT
                elif self.ts.ts_type == TS_STRAIGHT:
                    self.direction = DIR_STRAIGHT
                elif self.ts.ts_type == TS_STOP:
                    self.stop_registered = True
                elif self.ts.ts_type == TS_SPEED_0:
                    self.speed = SPEED_0
                elif self.ts.ts_type == TS_SPEED_1:
                    self.speed = SPEED_1
                elif self.ts.ts_type == TS_SPEED_2:
                    self.speed = SPEED_2
                elif self.ts.ts_type == TS_SPEED_3:
                    self.speed = SPEED_3
        self.ts.ts_type_last = self.ts.ts_type

    def identify_ts(self, img):
        mask_blue = cv2.inRange(img, LOWER_BLUE_HSV, UPPER_BLUE_HSV)
        mask_red = cv2.inRange(img, LOWER_RED_1_HSV, UPPER_RED_1_HSV) | cv2.inRange(img, LOWER_RED_2_HSV, UPPER_RED_2_HSV)
        if np.count_nonzero(mask_blue) > np.count_nonzero(mask_red):
            triangle_dir = self.get_triangle(mask_blue)
            if triangle_dir is not None:
                self.ts.ts_type = 4+triangle_dir
                return
        else:
            triangle_dir = self.get_triangle(mask_red)
            if triangle_dir is not None:
                self.ts.ts_type = triangle_dir
                return
        self.ts.ts_type = None

    def get_triangle(self, mask):
        """

        :param mask: the masked triangle
        :return:    0: left
                    1: right
                    2: up
                    3: down
        """
        up = None
        left = None
        cv2.imshow("Mask", mask)
        cv2.waitKey(1)
        if np.count_nonzero(mask) > 0.05*mask.size:
            cv2.dilate(mask, np.ones([5, 5], np.uint8), mask)
            _, contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            contour = max(contours, key=cv2.contourArea)
            _, triangle = cv2.minEnclosingTriangle(contour,)
            triangle = triangle.astype(int)
            # check if bounding triangle fits contour well enough
            if abs(1.0 - float(cv2.contourArea(triangle))/cv2.contourArea(contour)) > 0.3:
                return None
            x = triangle[:, 0, 0]
            y = triangle[:, 0, 1]
            (rows, cols) = mask.shape
            if min(x) < 0 or max(x) > cols or min(y) < 0 or max(y) > rows:
                return None
            x_diff = np.diff(x)
            y_diff = np.diff(y)
            min_x_diff = min([abs(x_diff[0]), abs(x_diff[1]), abs(x_diff[0]+x_diff[1])])
            min_y_diff = min([abs(y_diff[0]), abs(y_diff[1]), abs(y_diff[0]+y_diff[1])])

            # ts points left/right
            if min_x_diff < min_y_diff:
                # third point determines direction
                if abs(x_diff[0]) == min_x_diff:
                    point_idx = 2
                # first point determines direction
                elif abs(x_diff[1]) == min_x_diff:
                    point_idx = 0
                # second point determines direction:
                else:
                    point_idx = 1

                # points left
                left = x[point_idx] < x[point_idx-1]

            # ts points up/down
            else:
                if abs(y_diff[0]) == min_y_diff:
                    point_idx = 2
                elif abs(y_diff[1]) == min_y_diff:
                    point_idx = 0
                else:
                    point_idx = 1
                up = y[point_idx] < x[point_idx-1]
        if left is not None:
            if left:
                return 0
            return 1
        elif up is not None:
            if up:
                return 2
            return 3
        return None

    def identify_nn(self, img):
        img = cv2.resize(img, (60, 60))
        cv2.cvtColor(img, cv2.COLOR_HSV2RGB, img)
        img = img.reshape((1,) + img.shape)
        with self.graph.as_default():
            pred = self.nn_model.predict(img)
            if np.count_nonzero(pred) > 1:
                return
            pred_cls_idx = np.argmax(pred, axis=1)
            labels = self.train_gen.class_indices
            labels = dict((v,k) for k,v in labels.items())
            pred_out = labels[pred_cls_idx[0]]
            if pred_out == "links":
                self.ts.ts_type = TS_LEFT
            elif pred_out == "rechts":
                self.ts.ts_type = TS_RIGHT
            elif pred_out == "gerade":
                self.ts.ts_type = TS_STRAIGHT
            elif pred_out == "STOP":
                self.ts.ts_type = TS_STOP
            elif pred_out == "30":
                self.ts.ts_type = TS_SPEED_0
            elif pred_out == "50":
                self.ts.ts_type = TS_SPEED_1
            elif pred_out == "80":
                self.ts.ts_type = TS_SPEED_2
            elif pred_out == "100":
                self.ts.ts_type = TS_SPEED_3
            else:
                self.ts.ts_type = None


def create_model():
    img_width, img_height = 60, 60

    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)

    model = Sequential()
    model.add(Conv2D(64, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(8))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


def load_trained_model(path):
    model = create_model()
    model.load_weights(path)
    return model



def main():
    rospy.init_node("lane_follower_node")
    camera_settings = rospy.get_param("camera_settings")
    LaneFollower(width=camera_settings["width"], height=camera_settings["height"], nn=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        pass
    print("Shutting down")




if __name__ == '__main__':
    main()








