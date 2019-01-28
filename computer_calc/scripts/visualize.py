#!/usr/bin/env python
import rospy
import numpy as np
import cv2
from cv_bridge import CvBridge
from raspi_nodes.msg import LaneFollowerMsg
from sensor_msgs.msg import CompressedImage
import time
import os

TEXT_HEIGHT = 100

IMG_WIDTH = 1280
IMG_HEIGHT = 960 + TEXT_HEIGHT

DIR_LEFT = 0
DIR_RIGHT = 1
DIR_STRAIGHT = 2

TS_LEFT = 0
TS_RIGHT = 1
TS_STRAIGHT = 2
TS_STOP = 3
TS_SPEED_0 = 4
TS_SPEED_1 = 5
TS_SPEED_2 = 6
TS_SPEED_3 = 7


class Visualizer(object):
    def __init__(self, width, height, fps):
        os.chdir(os.path.dirname(os.getcwd()))
        os.chdir("pics2")
        self.counter = 0
        self.nrImages = 0
        self.input_width = width
        self.input_height = height
        self.fps = fps
        self.br = CvBridge()
        self.frame_counter = 0
        self.start = time.time()
        self.end = time.time()
        self.fps = 0
        self.camera_img = np.zeros([self.input_height, self.input_width, 3], dtype=np.uint8)
        self.sub_data = rospy.Subscriber(
            "/lane_follower/output",
            LaneFollowerMsg,
            self.data_rcv_cb,
            queue_size=1
        )
        self.sub_image = rospy.Subscriber(
            "/visualizer/camera_image",
            CompressedImage,
            self.img_rcv_cb,
            queue_size=1,
            buff_size=2**20
        )

    def img_rcv_cb(self, msg):
        self.camera_img = self.br.compressed_imgmsg_to_cv2(msg)

    def data_rcv_cb(self, msg):
        self.frame_counter = self.frame_counter + 1
        img = self.camera_img.copy()
        text_img = np.zeros((TEXT_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
        comb_img = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
        target_point = (msg.target_point_x, msg.target_point_y)
        lower_slice_row_idx = msg.lower_slice_row
        upper_slice_row_idx = msg.upper_slice_row
        slice_height = msg.slice_height
        lower_lane_boundaries = msg.lower_lane_boundaries
        upper_lane_boundaries = msg.upper_lane_boundaries
        crossing_detected = msg.crossing_detected
        ts_type = msg.ts_type
        speed = msg.speed
        direction = msg.direction
        (ts_x, ts_y, ts_w, ts_h) = (msg.ts_x, msg.ts_y, msg.ts_w, msg.ts_h)
        stop_registered = msg.stop_registered
        if self.frame_counter >= 30:
            self.frame_counter = 0
            self.end = time.time()
            self.fps = 30.0/(self.end-self.start)
            self.start = time.time()
        cv2.putText(text_img, "Crossing: {}".format(crossing_detected), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(text_img, "FPS: {:.2f}".format(self.fps), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1,
                    cv2.LINE_AA)
        cv2.putText(text_img, "DIR: {}".format(self.dir2string(direction)), (300, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(text_img, "spd: {}".format(speed), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(text_img, self.stop2string(stop_registered), (600, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.rectangle(img, (ts_x, ts_y), (ts_x+ts_w, ts_y+ts_h), (0, 255, 0), 2)
        if ts_w > 0 and (self.counter % 1 == 0):
            self.nrImages = self.nrImages + 1
            # print(self.nrImages)
            # cv2.imwrite("img_ts_dl{:04d}.jpg".format(self.nrImages), img[ts_y:ts_y+ts_h, ts_x:ts_x+ts_w])
        cv2.circle(img, target_point, 10, (255, 0, 0), 2)
        for i in lower_lane_boundaries:
            cv2.circle(img, (i, lower_slice_row_idx), 10, (255, 255, 0), 2)
        for i in upper_lane_boundaries:
            cv2.circle(img, (i, upper_slice_row_idx), 10, (0, 255, 255), 2)
        cv2.rectangle(img, (0, upper_slice_row_idx-slice_height/2), (640, upper_slice_row_idx+slice_height/2), (0, 0, 255), 2)
        cv2.rectangle(img, (0, lower_slice_row_idx - slice_height/2), (640, lower_slice_row_idx + slice_height/2), (0, 0, 255), 2)
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT-TEXT_HEIGHT))
        comb_img[0:TEXT_HEIGHT, :] = text_img[:]
        comb_img[TEXT_HEIGHT:, :] = img[:]
        cv2.imwrite("img_all_{:04d}.jpg".format(self.counter), comb_img)
        cv2.imshow("test", comb_img)
        cv2.waitKey(1)
        self.counter = self.counter+1

    def dir2string(self, direction):
        if direction == DIR_LEFT:
            return "LEFT"
        if direction == DIR_RIGHT:
            return "RIGHT"
        if direction == DIR_STRAIGHT:
            return "STRAIGHT"

    def stop2string(self, stop_registered):
        if stop_registered:
            return "STOP REGISTERED"
        return ""


def main():
    rospy.init_node("Visualizer")
    camera_settings = rospy.get_param("camera_settings")

    Visualizer(camera_settings["width"], camera_settings["height"], camera_settings["framerate"])

    try:
        rospy.spin()
    except KeyboardInterrupt:
        pass
    print("Shutting Down")


if __name__ == '__main__':
    main()
