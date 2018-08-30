#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
import cv2
import numpy as np
from cv_bridge import CvBridge


def callback(image_msg):
    cv_b = CvBridge()
    cv_image = cv_b.imgmsg_to_cv2(image_msg, desired_encoding="passthrough")
    cv_image_resized = cv2.resize(cv_image, (300, 300))
    np_image_resized = np.asarray(cv_image_resized)
    print('okay')


rospy.init_node('detect_object')
sub = rospy.Subscriber('image', Image, callback, queue_size=1)
rospy.spin()