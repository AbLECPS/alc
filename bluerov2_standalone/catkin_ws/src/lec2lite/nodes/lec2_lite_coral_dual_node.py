#!/usr/bin/env python

import numpy as np

import cv2
import os
import sys
import rospy, rospkg
import numpy as np
import math
import tf.transformations as trans
# import tensorflow as tf
import copy
import time
from threading import Lock  
from imutils import paths
from sensor_msgs.msg import Image
# from cv_bridge import CvBridge
from pycoral.utils import edgetpu
from pycoral.utils import dataset
from pycoral.adapters import common
from pycoral.adapters import classify


class LEC2Lite(object):
    def __init__(self):
        self.revmfunc = np.vectorize(self.label_to_image)
        self.lock = Lock()
        rp = rospkg.RosPack()
        tflite_filename_a = os.environ["ALC_WORKING_DIR"] + "/jupyter/admin_BlueROV/LEC2Lite/" + rospy.get_param('~filename', 'lec2_quant_edgetpu.tflite')
        tflite_filename_b = os.environ["ALC_WORKING_DIR"] + "/jupyter/admin_BlueROV/LEC2Lite/" + rospy.get_param('~filename', 'lec2_quant_edgetpu.tflite')

        print("LEC2Lite {} side model: {}".format("a", tflite_filename_a))
        print("LEC2Lite {} side model: {}".format("b", tflite_filename_b))
        
        self.lec2lite_l_pub = rospy.Publisher(
            '/vu_sss/lec2lite_l', Image, queue_size=1)    
        self.lec2lite_r_pub = rospy.Publisher(
            '/vu_sss/lec2lite_r', Image, queue_size=1)    
        
        self.sss_waterfall_l_sub = rospy.Subscriber(
            '/vu_sss/waterfall_l', Image, self.callback_sss_l, queue_size=1)
        self.sss_waterfall_r_sub = rospy.Subscriber(
            '/vu_sss/waterfall_r', Image, self.callback_sss_r, queue_size=1)
        
        # Load TFLite model and allocate tensors.
            
        self.interpreter_a = edgetpu.make_interpreter(tflite_filename_a)
        self.interpreter_b = edgetpu.make_interpreter(tflite_filename_b)

        try:
            self.interpreter_a.allocate_tensors() 
            self.interpreter_b.allocate_tensors() 
        except RuntimeError:
            pass 
        
        rospy.loginfo('VU LEC2 TFLite initialized for both sides')
        
    def callback_sss_l(self, msg):
        self.callback_sss(msg, self.lec2lite_l_pub, self.interpreter_a)

    def callback_sss_r(self, msg):
        self.callback_sss(msg, self.lec2lite_r_pub, self.interpreter_b)

    def callback_sss(self, msg, publisher, interpreter):
        '''
            TODO: Side dependent class - side & topic definition from launchfile
        '''
        start = time.time()
        # Get input and output tensors.
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Test model on random input data.
        input_shape = input_details[0]['shape']
        #self.lock.acquire()
        image = self.imgmsg_to_cv2(msg) 
        image = image.reshape((1,image.shape[0],image.shape[1],1))
        input_data = self.normalize(np.array(image, dtype=np.float32))
        interpreter.set_tensor(input_details[0]['index'], input_data)
        try:
            interpreter.invoke()  
        except RuntimeError:
            pass  
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]
        output_data = (output_data*255).astype(np.uint8)
        #  Blue channel represents seabed, correcting RGB color to match GT (black)
        output_data[:,:,0] *= 0
        #self.lock.release()
        img_msg = self.cv2_to_imgmsg(output_data)
        img_msg.header = msg.header
        publisher.publish(img_msg)
        rospy.loginfo('LEC2 TFLite output published for %s, computation time: %0.3fs' %(msg.header.frame_id, time.time() - start))

    def label_to_image(self, x1):
        if x1 > 0.5:
            return 255
        else:
            return 0

    def normalize(self, input_image):
        input_image = (np.float32(input_image)) / 255
        return  input_image
        

    def imgmsg_to_cv2(self, img_msg):
        '''
        https://answers.ros.org/question/350904/cv_bridge-throws-boost-import-error-in-python-3-and-ros-melodic/
            
            Provides conversions between OpenCV and ROS image formats in a hard-coded way.  
            CV_Bridge, the module usually responsible for doing this, is not compatible with Python 3,
            the language this all is written in.  So we create this module, and all is... well, all is not well,
            but all works.  :-/
        '''
        # if img_msg.encoding != "bgr8":
        #     rospy.logerr("This Coral detect node has been hardcoded to the 'bgr8' encoding.  Come change the code if you're actually trying to implement a new camera")
        dtype = np.dtype("uint8") # Hardcode to 8 bits...
        dtype = dtype.newbyteorder('>' if img_msg.is_bigendian else '<')

        image_opencv = np.ndarray(shape=(img_msg.height, img_msg.width, 1), # and three channels of data. Since OpenCV works with bgr natively, we don't need to reorder the channels.
                        dtype=dtype, buffer=img_msg.data)
        # If the byt order is different between the message and the system.
        if img_msg.is_bigendian == (sys.byteorder == 'little'):
            image_opencv = image_opencv.byteswap().newbyteorder()
        return image_opencv
    
    def cv2_to_imgmsg(self, cv_image):
        '''
        https://answers.ros.org/question/350904/cv_bridge-throws-boost-import-error-in-python-3-and-ros-melodic/
            
            Provides conversions between OpenCV and ROS image formats in a hard-coded way.  
            CV_Bridge, the module usually responsible for doing this, is not compatible with Python 3,
            the language this all is written in.  So we create this module, and all is... well, all is not well,
            but all works.  :-/
        '''
        img_msg = Image()
        img_msg.height = cv_image.shape[0]
        img_msg.width = cv_image.shape[1]
        img_msg.encoding = "bgr8"
        img_msg.is_bigendian = 0
        img_msg.data = cv_image.tobytes()
        img_msg.step = len(img_msg.data) // img_msg.height # That double line is actually integer division, not a comment
        return img_msg
        
if __name__ == "__main__":
    rospy.init_node("LEC2_TFLite" + rospy.get_param('~side', 'r'))
    lec2lite = LEC2Lite()
    rospy.spin()
