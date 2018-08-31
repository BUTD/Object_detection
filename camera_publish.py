#!/usr/bin/env python
""" This script is a ros node, that publishes a camera image as Image message named 'image' """
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass
import cv2
import time

# initialize ROS node and Publisher
rospy.init_node('VideoPublisher', anonymous=True)
VideoRaw = rospy.Publisher('image', Image, queue_size=1)

cam = cv2.VideoCapture('/dev/video1')

while not rospy.is_shutdown():
    # read the camera image, convert it to an Image Message and publish
    meta, frame = cam.read()
    msg_frame = CvBridge().cv2_to_imgmsg(frame)
    VideoRaw.publish(msg_frame)
    time.sleep(0.1)
