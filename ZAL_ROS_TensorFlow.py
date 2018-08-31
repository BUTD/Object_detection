#!/usr/bin/env python
""" This script is a ros node, which subscribes to an image and publishes home-made messages in the topic named
'bounding_boxes' """
import rospy
from zal_object_detection.msg import BoundingBox, BoundingBoxes
import cv2
import sys
from cv_bridge import CvBridge
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass
try:
    sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass
import tensorflow as tf
from sensor_msgs.msg import Image
import numpy as np


def zal_create_label_dict():
    """
    create a dictionary that maps class numbers to labels, read from a text file
    :return: dictionary with labels
    """
    coco_labels = {}
    with open('/home/nvidia/tf_trt_models/coco_labels') as f:
        for line_idx, line in enumerate(f.readlines()):
            coco_labels[line_idx] = line
    return coco_labels


def get_outputs(model_name='ssd_inception_v2_coco'):
    """
    Initialize the model, that means, load a saved model, get the output and input variables and create a tf session
    :param model_name:
    :return:
        :tf_sess: a tensorflow session
        :tf_input: input layer of the neural network
        :tf_scores: this is the output of the network that defines the probabilities for all predicted classes
        .tf_boxes: these are the coordinates of the
    """
    graph_def = tf.GraphDef()
    with open(model_name + '_trt.pb', "rb") as f:
        graph_def.ParseFromString(f.read())

    # create session an import graph
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_sess = tf.Session(config=tf_config)
    tf.import_graph_def(graph_def, name='')
    # get tensors that can be run for prediction
    tf_input = tf_sess.graph.get_tensor_by_name('input:0')
    tf_scores = tf_sess.graph.get_tensor_by_name('scores:0')
    tf_boxes = tf_sess.graph.get_tensor_by_name('boxes:0')
    tf_classes = tf_sess.graph.get_tensor_by_name('classes:0')

    return tf_sess, tf_input, tf_scores, tf_boxes, tf_classes


class ZAL_ROS_TensorFlow():
    def __init__(self):
        self._cv_bridge = CvBridge()

        self.x = tf.placeholder(tf.float32, [None, 28, 28, 1], name="x")
        self.keep_prob = tf.placeholder("float")
        self.tf_sess, self.tf_input, self.tf_scores, self.tf_boxes, self.tf_classes = get_outputs()

        self.sub = rospy.Subscriber('image', Image, self.callback, queue_size=1)
        self.pub = rospy.Publisher('bounding_boxes', BoundingBoxes, queue_size=1)
        self.threshold = 0.5
        self.labels = zal_create_label_dict()

    def callback(self, image_msg):
        cv_image = self._cv_bridge.imgmsg_to_cv2(image_msg, desired_encoding="passthrough")
        cv_image_resized = cv2.resize(cv_image, (300, 300))
        np_image_resized = np.asarray(cv_image_resized)
        scores, boxes, classes = self.tf_sess.run([self.tf_scores, self.tf_boxes, self.tf_classes], feed_dict={
            self.tf_input: np_image_resized[None, ...]})
        boxes = boxes[0]
        scores = scores[0]
        classes = classes[0]
        bb_msg = BoundingBoxes()
        bb_msg.bounding_boxes = self.result_to_bounding_box_msg(scores, boxes, classes, cv_image, self.threshold)
        self.pub.publish(bb_msg)

    def result_to_bounding_box_msg(self, scores, boxes, classes, image, threshold):
        bounding_boxes = []
        for i in range(len(scores)):
            if scores[i] > threshold:
                bounding_box_msg = BoundingBox()
                bounding_box_msg.Class = self.labels[classes[i]]
                bounding_box_msg.probability = scores[i]
                box = boxes[i] * np.array([image.shape[0], image.shape[1], image.shape[0], image.shape[1]])
                bounding_box_msg.xmin = min(box[1], box[3])
                bounding_box_msg.ymin = min(box[0], box[2])
                bounding_box_msg.xmax = max(box[1], box[3])
                bounding_box_msg.ymax = max(box[0], box[2])
                bounding_boxes.append(bounding_box_msg)
        return bounding_boxes

    def main(self):
        rospy.spin()


if __name__ == '__main__':
    rospy.init_node('detect_objects')
    tensor = ZAL_ROS_TensorFlow()
    tensor.main()
