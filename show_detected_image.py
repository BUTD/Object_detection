#!/usr/bin/env python
""" This script subscribes to a published image and to detected bounding boxes and shows the resulting image"""
import rospy
from zal_object_detection.msg import BoundingBox, BoundingBoxes
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import message_filters


class ShowImage():
    def __init__(self):
        self.cv_bridge = CvBridge()
        rospy.on_shutdown(self.cleanup)
        rospy.init_node('show_detected_image')
        # cv2.namedWindow('live', cv2.WINDOW_NORMAL)
        sub_bb = message_filters.Subscriber('bounding_boxes', BoundingBoxes)
        sub_im = message_filters.Subscriber('image', Image)
        ts = message_filters.TimeSynchronizer([sub_bb, sub_im], 5)
        ts.registerCallback(self.callback)

    def callback(self, bounding_box_msg, img_msg):
        assert bounding_box_msg.header.stamp == img_msg.header.stamp
        print(bounding_box_msg)
        cv_image = self.cv_bridge.imgmsg_to_cv2(img_msg, desired_encoding="passthrough")
        if bounding_box_msg:
            for bb in bounding_box_msg.bounding_boxes:
                cv2.rectangle(cv_image, (bb.xmin, bb.ymin), (bb.xmax, bb.ymax), (255, 255, 0), 3)
        cv2.imshow('live', cv_image)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            rospy.signal_shutdown('Stop')
        print('Wait')

    def cleanup(self):
        print("Shutting down vision node.")
        cv2.destroyAllWindows()


def main():
    try:
        ShowImage()
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down vision node.")
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
