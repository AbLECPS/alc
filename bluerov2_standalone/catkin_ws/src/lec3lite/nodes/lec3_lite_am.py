#!/usr/bin/env python

import rospy
import numpy as np
import math
import collections
import warnings

import time
import warnings
import copy
from std_msgs.msg import Float32MultiArray


# ALC TOOLCHAIN - VAE AM
#removing any gpu usage
import rospkg
import os
# os.environ["CUDA_VISIBLE_DEVICES"]=""

#forcing to use the alc_utils in the cpu_test folder
import sys
from datetime import datetime 

import alc_utils.assurance_monitor
import pickle
from torchvision import transforms
import torch
import threading

class LEC3LiteAM(object):
    def __init__(self):
        self.namespace = rospy.get_namespace().replace('/', '')
        self.am_type = rospy.get_param('~am_type', 'vae')
        assert self.am_type in ['vae', 'svdd'], "Not supported AM type"

        self.am_lec3lite_pub = rospy.Publisher(
            '/lec3lite/am_' + self.am_type, 
            Float32MultiArray, 
            queue_size=1) 

        ###### ALC VAE AM
        #use the am in the current folder
        rospack = rospkg.RosPack()
        am_path = os.environ["ALC_WORKING_DIR"] + rospy.get_param('~path', "/jupyter/admin_BlueROV/LEC3Lite_AM/vae/assurance_monitor_2022-11-22_18-37-53")
        print("##########################################\n\n\n")
        print("AM Path:")
        print(am_path)
        print("\n\n\n##########################################")
        _ams = alc_utils.assurance_monitor.load_assurance_monitor("multi")
        _ams.load([am_path])
        self.am = _ams.assurance_monitors[0]
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

        self.fls_bins_sub = rospy.Subscriber(
            '/vu_fls/bins', Float32MultiArray, self.callback_fls, queue_size=1)

    def get_am(self, x):
        return self.am.evaluate(x, None)

    def callback_fls(self, msg):
        t = threading.Thread(target=self.am_thread, args=(msg,))
        t.start()

    def am_thread(self, msg):
        start_time = datetime.now() 
        am_output = self.get_am(self.normalize(msg.data)) 
        time_elapsed = datetime.now() - start_time 
        am_output = am_output[0]
        rospy.loginfo(" >> LEC3 {}-AM \tlog(m): {}, det: {} Time elapsed (hh:mm:ss.ms) {}".format(self.am_type, am_output[-2], am_output[-1], time_elapsed))
        am_msg = Float32MultiArray()
        am_msg.data = am_output        
        self.am_lec3lite_pub.publish(am_msg)
    
    def normalize(self, x, normalizer=255.0):        
        # Normalizing to [0,1]
        x = np.array(x) / normalizer
        # masking the end of the range
        x[-10:] = 0
        return  x

if __name__=='__main__':
    rospy.init_node('lec3lite_am', log_level=rospy.ERROR)
    try:
        node = LEC3LiteAM()
        rospy.spin()
    except rospy.ROSInterruptException:
        print('caught exception')
