#!/usr/bin/env python3
import select
import sys
import termios
import tty

import rospy
from geometry_msgs.msg import Vector3
from rm_radar_msgs.msg import LaserCameraCalib


class BiasController:
    def __init__(self):
        self.bias_topic = rospy.get_param("~bias_topic", "/laser_camera_calibrate/bias")
        self.manual_bias_topic = rospy.get_param("~manual_bias_topic", "/laser_camera_calibrate/manual_bias")
        self.bias_x = float(rospy.get_param("~init_bias_x", 0.0))
        self.bias_y = float(rospy.get_param("~init_bias_y", 0.0))
        self.step_x = float(rospy.get_param("~step_x", 1.0))
        self.step_y = float(rospy.get_param("~step_y", 1.0))
        self.step_scale = float(rospy.get_param("~step_scale", 2.0))
        self.step_min = float(rospy.get_param("~step_min", 0.01))
        self.step_max = float(rospy.get_param("~step_max", 100.0))
        self.poll_hz = float(rospy.get_param("~poll_hz", 50.0))

        self.default_bias_x = self.bias_x
        self.default_bias_y = self.bias_y

        self.bias_pub = rospy.Publisher(self.bias_topic, LaserCameraCalib, queue_size=20, latch=True)
        self.manual_bias_pub = rospy.Publisher(self.manual_bias_topic, Vector3, queue_size=20)
        self.publish_bias()

        rospy.loginfo("Publishing bias to topic: %s", self.bias_topic)
        rospy.loginfo("Manual sample trigger topic: %s", self.manual_bias_topic)
        self.print_help()
        self.log_state()

    def print_help(self):
        print("")
        print("=== Laser-Camera Bias WASD ===")
        print("w/s: bias_y -/+")
        print("a/d: bias_x -/+")
        print("z/x: step /2 or *2")
        print("space: confirm hit and sample current bias")
        print("r: reset to init bias")
        print("q: quit")
        print("============================")
        print("")

    def publish_bias(self):
        msg = LaserCameraCalib()
        msg.header.stamp = rospy.Time.now()
        msg.armor_valid = False
        msg.armor_confidence = 0.0
        msg.armor_h_px = 0.0
        msg.bias_valid = True
        msg.bias_x = self.bias_x
        msg.bias_y = self.bias_y
        self.bias_pub.publish(msg)

    def log_state(self):
        rospy.loginfo("bias_x=%.3f bias_y=%.3f step_x=%.3f step_y=%.3f",
                      self.bias_x, self.bias_y, self.step_x, self.step_y)

    def publish_manual_sample(self):
        msg = Vector3()
        msg.x = self.bias_x
        msg.y = self.bias_y
        msg.z = 0.0
        self.manual_bias_pub.publish(msg)
        rospy.loginfo("sample trigger: bias_x=%.3f bias_y=%.3f", self.bias_x, self.bias_y)

    def get_key(self, timeout):
        rlist, _, _ = select.select([sys.stdin], [], [], timeout)
        if rlist:
            return sys.stdin.read(1)
        return ""

    def run(self):
        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setraw(sys.stdin.fileno())
            rate = rospy.Rate(max(1.0, self.poll_hz))
            while not rospy.is_shutdown():
                key = self.get_key(1.0 / max(1.0, self.poll_hz))
                if not key:
                    rate.sleep()
                    continue

                changed = False
                if key == "w":
                    self.bias_y -= self.step_y
                    changed = True
                elif key == "s":
                    self.bias_y += self.step_y
                    changed = True
                elif key == "a":
                    self.bias_x -= self.step_x
                    changed = True
                elif key == "d":
                    self.bias_x += self.step_x
                    changed = True
                elif key == "z":
                    self.step_x = max(self.step_min, self.step_x / self.step_scale)
                    self.step_y = max(self.step_min, self.step_y / self.step_scale)
                    rospy.loginfo("step -> step_x=%.3f step_y=%.3f", self.step_x, self.step_y)
                elif key == "x":
                    self.step_x = min(self.step_max, self.step_x * self.step_scale)
                    self.step_y = min(self.step_max, self.step_y * self.step_scale)
                    rospy.loginfo("step -> step_x=%.3f step_y=%.3f", self.step_x, self.step_y)
                elif key == "r":
                    self.bias_x = self.default_bias_x
                    self.bias_y = self.default_bias_y
                    changed = True
                elif key == " ":
                    self.publish_manual_sample()
                elif key == "q":
                    rospy.loginfo("Quit bias keyboard.")
                    break

                if changed:
                    self.publish_bias()
                    self.log_state()

                rate.sleep()
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


if __name__ == "__main__":
    rospy.init_node("bias_controller")
    node = BiasController()
    node.run()
