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
from collections import deque
from threading import Lock  
from imutils import paths
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


class LEC3Lite(object):
    def __init__(self):
        self.lec3lite_pub_bins = rospy.Publisher(
            '/lec3lite/bins', Float32MultiArray, queue_size=1)    
        
        self.lec3lite_pub_ranges = rospy.Publisher(
            '/lec3lite/ranges', Float32MultiArray, queue_size=1)
        
        self.lec3lite_pub_waterfall = rospy.Publisher(
            '/lec3lite/waterfall', Image, queue_size=1)    
        
        self.fls_bins_sub = rospy.Subscriber(
            '/vu_fls/bins', Float32MultiArray, self.callback_fls, queue_size=1)
        
                # Load TFLite model and allocate tensors.
        rp = rospkg.RosPack()
        filename = os.environ["ALC_WORKING_DIR"] + "/jupyter/admin_BlueROV/LEC3Lite/" + rospy.get_param('~filename', 'best_model/lec3_quant.tflite')
        
        physical_devices = tf.config.list_physical_devices('GPU')
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        except:
            pass
            
        self.softmax_threshold = 0.1  
        physical_devices = tf.config.list_physical_devices('GPU') 
        for gpu_instance in physical_devices: 
            tf.config.experimental.set_memory_growth(gpu_instance, True) 
        self.interpreter = tf.lite.Interpreter(model_path=filename)
        self.cvbridge = CvBridge()

        self.fls_bins = int(rospy.get_param('~num_range_bins', 252))
        self.fls_range = int(rospy.get_param('~fls_range', 30))
        self.use_hw_fls = rospy.get_param('~use_hw_fls', False)
        if self.use_hw_fls:
            self.fls_lines = 25
        else:
            self.fls_lines = rospy.get_param('~fls_lines', 100)

        self.fls_waterfall = deque(maxlen=self.fls_lines)
        
                
        try:
            self.interpreter.allocate_tensors() 
        except RuntimeError:
            pass 
        
        rospy.loginfo('VU LEC3 TFLite initialized')
        
    def callback_fls(self, msg):
        start = time.time()
        # Get input and output tensors.
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Test model on random input data.
        self.input_shape = self.input_details[0]['shape']
        # self.lock.acquire()
        input_data = self.normalize(msg.data)  
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        try:
            self.interpreter.invoke()  
        except RuntimeError:
            pass  
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

        output_data[output_data<self.softmax_threshold] = 0
        lec_ind = np.where(output_data[:]>0)
        ranges = np.array(lec_ind) / 252.0 * 30

        

        if np.shape(ranges)[1] == 0:
            ranges = np.array(np.inf)
        
        msg = Float32MultiArray()
        msg.data = ranges.flatten()
        self.lec3lite_pub_ranges.publish(msg)

        msg.data = output_data.flatten()
        self.lec3lite_pub_bins.publish(msg)
        
        self.fls_waterfall.append(self.denormalize(output_data))

        if len(self.fls_waterfall) == self.fls_lines:
            # waterfall_msg = Image()
            # waterfall_msg.width = self.fls_bins
            # waterfall_msg.height = self.fls_lines
            # waterfall_msg.header.stamp = rospy.Time.now()
            img = np.array(self.fls_waterfall, dtype=np.uint8)
            # Flip H and V
            # Top = newest, left = closest
            img = np.flip(img, 0)
            waterfall_msg = self.cvbridge.cv2_to_imgmsg(img, encoding="mono8")
            self.lec3lite_pub_waterfall.publish(waterfall_msg)

        rospy.loginfo('LEC3 TFLite output published, computation time: %0.3fs' %(time.time() - start))

    def denormalize(self, lec_output):
        lec_output = np.where(lec_output > self.softmax_threshold, 1, 0)
        return lec_output*255

    def normalize(self, lec_input, normalizer=255.0):        
        # Normalizing to [0,1]
        lec_input = np.array(lec_input) / normalizer
        # masking the end of the range
        lec_input[-15:] = 0
        # Typecast to tf.float32
        lec_input=lec_input.reshape((1,lec_input.shape[0]))
        return  tf.cast(lec_input, tf.float32)
       
if __name__ == "__main__":
    rospy.init_node("LEC3_TFLite", log_level=rospy.ERROR)
    lec3lite = LEC3Lite()
    rospy.spin()
