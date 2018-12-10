#!/usr/bin/env python
import rospy
import sys
import math
from std_msgs.msg import Int16, Float32
from geometry_msgs.msg import PointStamped

ROSPY_RATE = 20
FORWARD_THRESHOLD = 700
BACKWARD_THRESHOLD = 800
MAX_THROTTLE = 4000
MIN_THROTTLE = -4000
MIN_STEERING = -20
MAX_STEERING = 20

class Controller(object):

    def __init__(self):
        self.rate = rospy.Rate(ROSPY_RATE)
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
        self.subscriber = rospy.Subscriber(
            "/tracking_point/PointStamped",
            PointStamped,
            self.point_rcv_callback,
            queue_size=1
        )
        self.throttle = 0
        self.steering = 0.0
        self.desired_distance = 500
        self.distance = self.desired_distance
        self.desired_angle = 0.0
        self.angle = self.desired_angle

        self.pid_throttle = PID(K_P=10, K_I=0.0, K_D=1, threshold=10)
        self.pid_steering = PID(K_P=1, threshold=0)
        self.funnel_throttle = Funnel()

    def run(self):
        msg_throttle = Int16()
        msg_steering = Float32()
        # run this loop forever
        while not rospy.is_shutdown():
            # calculate controller output of the throttle controller
            self.throttle =self.pid_throttle.execute(
                self.desired_distance, self.distance, 1.0/ROSPY_RATE)
            #self.throttle = self.funnel_throttle.execute(self.desired_distance, self.distance)

            print("Throttle: {}".format(self.throttle))
            # add thresholds. smaller values wouldnt make the car move
            if (self.throttle > 0):
                self.throttle = self.throttle + FORWARD_THRESHOLD
            elif (self.throttle < 0):
                self.throttle = self.throttle - BACKWARD_THRESHOLD

            # check for throttle limits
            if self.throttle > MAX_THROTTLE:
                self.throttle = MAX_THROTTLE
            elif self.throttle < MIN_THROTTLE:
                self.throttle = MIN_THROTTLE

            # publish throttle message
            msg_throttle.data = int(self.throttle)
            self.throttle_pub.publish(msg_throttle)

            # only steer when driving forward
            if self.throttle > 0:
                # calculate controller output of the steering angle controller
                self.steering = self.pid_steering.execute(
                    self.desired_angle, self.angle, 1.0/ROSPY_RATE
                )

                # check for angle limits
                if self.steering > MAX_STEERING:
                    self.steering = MAX_STEERING
                elif self.steering < MIN_STEERING:
                    self.steering = MIN_STEERING
            else:
                self.steering = 0

            # publish steering angle message
            msg_steering.data = float(self.steering)
            self.angle_pub.publish(msg_steering)

            # sleep for the remaining period time
            self.rate.sleep()

    def point_rcv_callback(self, msg):
        point = (msg.point.x, msg.point.y, msg.point.z)
        self.distance = point[2]
        # calculate the angle to the tracking point in degree
        self.angle = math.atan(point[0]/point[2])/math.pi*180

    def on_shutdown_callback(self):
        msg_throttle = Int16()
        msg_steering = Float32()
        msg_throttle.data = 0
        msg_steering.data = 0.0
        self.angle_pub.publish(msg_steering)
        self.throttle_pub.publish(msg_throttle)



class PID(object):
    def __init__(self, K_P=1, K_I=0, K_D=0, threshold=100):
        self.K_P = K_P
        self.K_I = K_I
        self.K_D = K_D
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

class Funnel(object):
    def __init__(self):
        self.time = 0.0
    def execute(self, desired_value, actual_value):
        fun_value = self.funnel_fct(self.time)
        error = actual_value - desired_value
        output = 5.0/(1.0-fun_value*fun_value*error*error)*error
        self.time = self.time + 1.0/ROSPY_RATE
        return output
    def funnel_fct(self, x):
        return math.e**(-100*x) * 10**15





def main():
    rospy.init_node('controller_node', anonymous=True)
    controller = Controller()
    try:
        controller.run()
    except rospy.ROSInterruptException:
        print("Shutting down controller_node")

if __name__ == '__main__':
    main()
