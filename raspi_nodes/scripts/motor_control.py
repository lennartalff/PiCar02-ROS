#!/usr/bin/env python
import rospy
import Adafruit_PCA9685
from std_msgs.msg import Float32, Int16

SERVO_CHANNEL = 6
MOTOR_FORWARD_CHANNEL = 11
MOTOR_BACKWARD_CHANNEL = 10
IN_1_CHANNEL = 8    # BACKWARD
IN_2_CHANNEL = 9    # FORWARD

ANGLE_MAX = 20
ANGLE_MIN = -20
THROTTLE_MIN = -4095
THROTTLE_MAX = 4095


class MotorController(object):
    def __init__(self, freq=200, address=0x40):
        rospy.on_shutdown(self.on_shutdown_callback)
        self.subscriber_steering = rospy.Subscriber(
            "/motor_control/steering",
            Float32,
            self.steering_callback,
            queue_size=1)
        self.Subscriber_throttle = rospy.Subscriber(
            "/motor_control/throttle",
            Int16,
            self.throttle_callback,
            queue_size=1)
        self.pwm = Adafruit_PCA9685.PCA9685(address=address)
        self.pwm.set_pwm_freq(freq)
        self.pwm.set_pwm(IN_1_CHANNEL, 0, 4000)
        self.pwm.set_pwm(IN_2_CHANNEL, 0, 4000)
        self.pwm.set_pwm(SERVO_CHANNEL, 0, 1228)

    def set_steering_angle(self, angle):
        # PWM MAX: 3225
        # PWM MIN: 2100
        if angle > ANGLE_MAX:
            angle = ANGLE_MAX
        if angle < ANGLE_MIN:
            angle = ANGLE_MIN
        self.pwm.set_pwm(SERVO_CHANNEL, 0, int(1228+(angle+1.14)/0.0755))
        print("Servo PWM: {}".format(int(1228+25*angle)))

    def set_throttle(self, throttle):
        if throttle > THROTTLE_MAX:
            throttle = THROTTLE_MAX
        if throttle < THROTTLE_MIN:
            throttle = THROTTLE_MIN
        print("Throttle PWM: {}".format(int(throttle)))
        if throttle < 0:
            self.pwm.set_pwm(MOTOR_FORWARD_CHANNEL, 0, 0)
            self.pwm.set_pwm(MOTOR_BACKWARD_CHANNEL, 0, -1*throttle)
        else:
            self.pwm.set_pwm(MOTOR_BACKWARD_CHANNEL, 0, 0)
            self.pwm.set_pwm(MOTOR_FORWARD_CHANNEL, 0, throttle)

    def steering_callback(self, msg):
        print("Received desired steering angle: {}".format(msg.data))
        self.set_steering_angle(msg.data)

    def throttle_callback(self, msg):
        self.set_throttle(msg.data)

    def on_shutdown_callback(self):
        self.set_throttle(0)
        self.set_steering_angle(0.0)


def main():
    rospy.init_node('motor_control_node', anonymous=True)
    motor_controller = MotorController()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print('Shutting down motor_control_node')


if __name__ == '__main__':
    main()
