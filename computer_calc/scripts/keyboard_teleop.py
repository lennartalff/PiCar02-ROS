#!/usr/bin/env python

import rospy
import keyboard
from std_msgs.msg import Float32, Int16

ROSPY_RATE = 30

class RemoteController(object):
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


    def run(self):
        while not rospy.is_shutdown():





def main():
    rospy.init_node('keyboard_teleop_node', anonymous=True)
    rcController = RemoteController()
    try:
        rcController.run()
    except rospy.ROSInterruptException:
        print("Shutting down keyboard_teleop_node")

if __name__ == '__main__':
    main()
