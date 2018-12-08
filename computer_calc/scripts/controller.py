#!/usr/bin/env python
import rospy
import sys
from std_msgs.msg import Int16, Float32
from geometry_msgs.msg import PointStamped

ROSPY_RATE = 20
K_P = 3
K_I = 0
K_D = 0

class Controller(object):

    def __init__(self):
        self.rate = rospy.Rate(ROSPY_RATE)
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
        self.subscriber = rospy.Subscriber(
            "/tracking_point/PointStamped",
            PointStamped,
            self.point_rcv_callback,
            queue_size=1
        )
        self.error_prior = 0
        self.integral = 0
        self.throttle = 0
        self.distance = 1000
        self.desired_value = 1000

    def run(self):
        try:
            msg = Int16()
            while not rospy.is_shutdown():
                print("throttle: {}".format(self.throttle))
                self.throttle =self.pid(
                    self.desired_value, self.distance, 0.05)
                if (self.throttle > 0):
                    self.throttle = self.throttle + 800
                elif (self.throttle < 0):
                    self.throttle = self.throttle - 850
                if self.throttle > 4000:
                    self.throttle = 4000
                elif self.throttle < -4000:
                    self.throttle = -4000
                msg.data = int(self.throttle)
                self.throttle_pub.publish(msg)
                self.rate.sleep()
        except KeyboardInterrupt:
            sys.exit()

    def pid(self, desired_value, actual_value, delta_t):

        if abs(desired_value-actual_value) > 0.01:
            err = -desired_value + actual_value
            self.integral = self.integral + err*delta_t
            derivate = (err - self.error_prior)/delta_t
            output = K_P*err + K_I*self.integral + K_D*derivate
            self.error_prior=err
        else:
            output = 0
        return output

    def point_rcv_callback(self, msg):
        point = (msg.point.x, msg.point.y, msg.point.z)
        self.distance = point[2]







def main():
    rospy.init_node('controller_node', anonymous=True)
    controller = Controller()
    try:
        controller.run()
    except rospy.ROSInterruptException:
        print("Shutting down controller_node")

if __name__ == '__main__':
    main()
