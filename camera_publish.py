#!/usr/bin/env python
""" This script is a ros node, that publishes a camera image as Image message named 'image' """
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import UInt8
from zal_object_detection.msg import UInts_zal


from cv_bridge import CvBridge, CvBridgeError
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass
import cv2
import time
from rospy.numpy_msg import numpy_msg
import numpy as np

rospy.init_node('VideoPublisher', anonymous=True)

VideoRaw = rospy.Publisher('image', Image, queue_size=1)
# VideoRaw = rospy.Publisher('image_np', numpy_msg(UInts_zal), queue_size=1)

cam = cv2.VideoCapture('/dev/video1')

while not rospy.is_shutdown():
    meta, frame = cam.read()

    msg_frame = CvBridge().cv2_to_imgmsg(frame)

    # msg_frame = np.asarray(frame, dtype=np.uint8)

    # print(np.ndarray.flatten(msg_frame).shape)
    VideoRaw.publish(msg_frame)
    time.sleep(0.1)
