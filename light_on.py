#!/usr/bin/env python
""" This script is a ros node, which subscribes to Bounding boxes and publishes publishes a signal if a person is
detected in one of 5 preceding frames """

import rospy
import numpy as np
from zalamander_msgs.msg import BoundingBoxes, BoundingBox
from std_msgs.msg import Bool


class Light():
    def __init__(self):
        self.predictions = [0, 0, 0, 0, 0]
        self.pub = rospy.Publisher('light_on', Bool, queue_size=1)
        rospy.Subscriber('bounding_boxes', BoundingBoxes, self.callback, queue_size=1)

    def callback(self, bounding_box_msg):
        """
        if in at least two of the last 5 frames there was a person, then return True
        :param bounding_box_msg:
        :return:
        """
        person_detected = False
        for bounding_box in bounding_box_msg.bounding_boxes:
            if bounding_box.Class == 'person':
                person_detected = True
        if person_detected:
            self.predictions.append(1)
        else:
            self.predictions.append(0)
        self.predictions.pop(0)
        if sum(self.predictions) > 1:
            self.pub.publish(True)
        else:
            self.pub.publish(False)

    def main(self):
        rospy.spin()


if __name__ == '__main__':
    rospy.init_node('turn_light_on')
    light = Light()
    light.main()

