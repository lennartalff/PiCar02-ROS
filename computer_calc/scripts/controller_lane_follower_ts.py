#!/usr/bin/env python
import rospy
import math
from std_msgs.msg import Int16, Float32
import time

MAX_THROTTLE = 4000
MIN_THROTTLE = -4000
MIN_STEERING = -20.0
MAX_STEERING = 20.0


class Controller(object):
    def __init__(self):
        rospy.on_shutdown(self.on_shutdown_callback)
        self.angle_pub = rospy.Publisher(
            "/motor_control/steering",
            Float32,
            queue_size=1
        )
        self.throttle_pub = rospy.Publisher(
            "/motor_control/throttle",
            Int16,
            queue_size=1
        )
        self.point_sub = rospy.Subscriber(
            "/lane_follower/normalized_control_error",
            Float32,
            self.ctrl_err_rcv_callback,
            queue_size=1
        )
        self.throttle_sub = rospy.Subscriber(
            "/lane_follower/target_speed",
            Int16,
            self.throttle_rcv_callback,
            queue_size=1
        )
        self.throttle = 0
        self.steering = 0.0
        self.desired_angle = 0.0
        self.angle = self.desired_angle
        self.stop = False
        self.pid_steering = PID(k_p=5.0, k_d=0, threshold=0)

        self.old_time = time.time()
        self.new_time = time.time()

    def ctrl_err_rcv_callback(self, msg):
        err = msg.data
        self.new_time = time.time()
        if abs(err) > 1.0:
            self.stop = True
            self.throttle = 0
            self.steering = 0
        else:
            self.stop = False
            self.angle = MAX_STEERING * err
            if not (self.throttle == 0):
                self.steering = self.pid_steering.execute(
                    self.desired_angle, self.angle, self.new_time-self.old_time
                )
            else:
                self.steering = 0

        msg = Float32()
        msg.data = self.steering
        self.angle_pub.publish(msg)
        msg = Int16()
        msg.data = self.throttle
        self.throttle_pub.publish(msg)
        self.old_time = self.new_time

    def throttle_rcv_callback(self, msg):
        if not self.stop:
            self.throttle = msg.data
        else:
            self.throttle = 0

    def on_shutdown_callback(self):
        msg_throttle = Int16()
        msg_steering = Float32()
        msg_throttle.data = 0
        msg_steering.data = 0.0
        self.angle_pub.publish(msg_steering)
        self.throttle_pub.publish(msg_throttle)


class PID(object):
    def __init__(self, k_p=1.0, k_i=0.0, k_d=0.0, threshold=100):
        self.K_P = k_p
        self.K_I = k_i
        self.K_D = k_d
        self.integral = 0
        self.error_prior = 0
        self.threshold = threshold

    def execute(self, desired_value, actual_value, delta_t):
        err = -desired_value + actual_value
        if abs(err) < self.threshold:
            err = 0
        self.integral = self.integral + err*delta_t
        derivate = (err - self.error_prior)/delta_t
        output = self.K_P*err + self.K_I*self.integral + self.K_D*derivate
        self.error_prior = err
        return output

def main():
    rospy.init_node("controller_node")
    Controller()
    rospy.spin()
    print("Shutting down controller_node")


if __name__ == '__main__':
    main()
