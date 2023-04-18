#!/usr/bin/env python

import numpy as np

import cv2
import os
import sys
import rospy, rospkg
import numpy as np
import math
import tensorflow as tf
import copy
import time
from threading import Lock  
from imutils import paths
from sensor_msgs.msg import Image
# from cv_bridge import CvBridge
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

class LEC2Lite(object):
    def __init__(self):
        self.revmfunc = np.vectorize(self.label_to_image)
        self.lock = Lock()
        self.side = rospy.get_param('~side', 'r')
        self.lec2lite_pub = rospy.Publisher(
            '/vu_sss/lec2lite_' + self.side, Image, queue_size=1)    
        
        self.sss_waterfall_sub = rospy.Subscriber(
            '/vu_sss/waterfall_' + self.side, Image, self.callback_sss, queue_size=1)
        
        physical_devices = tf.config.list_physical_devices('GPU')
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        except:
            pass
 
        # Load TFLite model and allocate tensors.
        rp = rospkg.RosPack()
        filename = os.environ["ALC_WORKING_DIR"] + "/jupyter/admin_BlueROV/LEC2Lite/" + rospy.get_param('~filename', 'lec2.tflite')
        # filename = rp.get_path("lec2lite") + "/nodes/lec2_quant.tflite"

        #config = tf.ConfigProto()
        #config.gpu_options.allow_growth = True
        #sess = tf.Session(config=config)
        physical_devices = tf.config.list_physical_devices('GPU') 
        for gpu_instance in physical_devices: 
            tf.config.experimental.set_memory_growth(gpu_instance, True)
        self.interpreter = tf.lite.Interpreter(model_path=filename)
                
        try:
            self.interpreter.allocate_tensors() 
        except RuntimeError:
            pass 
        
        rospy.loginfo('VU LEC2 TFLite initialized for "%s" side' %(self.side))
        
    def callback_sss(self, msg):
        '''
            TODO: Side dependent class - side & topic definition from launchfile
        '''
        start = time.time()

        # Get input and output tensors.
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Test model on random input data.
        self.input_shape = self.input_details[0]['shape']
        self.lock.acquire()
        image = self.imgmsg_to_cv2(msg) 
        image = image.reshape((1,image.shape[0],image.shape[1],1))
        input_data = self.normalize(np.array(image, dtype=np.float32))
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        try:
            self.interpreter.invoke()  
        except RuntimeError:
            pass  
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
                
        output_data = (output_data*255).astype(np.uint8)        
        output_data[:,:,0] = output_data[:,:,2]
        output_data[:,:,1] = output_data[:,:,2] 
        self.lock.release()
        img_msg = self.cv2_to_imgmsg(output_data)
        img_msg.header = msg.header
        self.lec2lite_pub.publish(img_msg)

        rospy.loginfo('LEC2 TFLite output published for %s, computation time: %0.3fs' %(msg.header.frame_id, time.time() - start))


    def label_to_image(self, x1):
        if x1 > 0.5:
            return 255
        else:
            return 0

    def normalize(self, input_image):
        input_image = (np.float32(input_image)) / 255
        return  tf.cast(input_image, tf.float32)

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
    rospy.init_node("LEC2_TFLite" + rospy.get_param('~side', 'r'), log_level=rospy.ERROR)
    lec2lite = LEC2Lite()
    rospy.spin()
