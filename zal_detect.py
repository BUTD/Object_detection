#!/usr/bin/env python
import rospy
import cv2
from cv_bridge import CvBridge
from std_msgs.msg import String, Float32
import tensorflow as tf
import numpy as np


model_name = 'ssd_inception_v2_coco'
data_dir = './data/'
config_file = model_name + '.config'
checkpoint_file = 'model.ckpt'
image_path = './data/donald.jpg'

# load the graph from memory
graph_def = tf.GraphDef()
with open('ssd_' + model_name + '_trt.pb', "rb") as f:
    graph_def.ParseFromString(f.read())


# create session an import graph
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf.import_graph_def(graph_def, name='')

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

rospy.init_node('detect_object', anonymous=True)
# These should be combined into a single message
pub = rospy.Publisher('object_detected', String, queue_size = 1)
pub1 = rospy.Publisher('object_detected_probability', Float32, queue_size = 1)
bridge = CvBridge()

msg_string = String()
msg_float = Float32()
tf_sess = tf.Session()

def zal_predict_on_image(image_resized, tf_sess, tf_scores, tf_boxes, tf_classes):
    scores, boxes, classes = tf_sess.run([tf_scores, tf_boxes, tf_classes], feed_dict={
        tf_input: image_resized[None, ...])
    boxes = boxes[0]
    scores = scores[0]
    classes = classes[0]

    return boxes, scores, classes

def callback(image_msg):
    # First convert the image to OpenCV image
    cv_image = bridge.imgmsg_to_cv2(image_msg, desired_encoding="passthrough")
    cv_image_resized = cv2.resize(cv_image, (300, 300))  # resize image
    np_image_resized = np.asarray(cv_image_resized)  # read as np array
    boxes, scores, classes = zal_predict_on_image(np_image_resized, t)





    global graph  # This is a workaround for asynchronous execution
    with graph.as_default():
 # !!!!!       preds = model.predict(np_image)  # Classify the image
        # decode returns a list  of tuples [(class,description,probability),(class, descrip ...
        pred_string = decode_predictions(preds, top=1)[0]  # Decode top 1 predictions
        msg_string.data = pred_string[0][1]
        msg_float.data = float(pred_string[0][2])
        pub.publish(msg_string)
        pub1.publish(msg_float)


rospy.Subscriber("camera/image_raw", Image, callback, queue_size=1, buff_size=16777216)

while not rospy.is_shutdown():
    rospy.spin()
