#!/usr/bin/env python

import rospy
import numpy as np
import math
import collections
import cv2
import warnings

from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import Range
from geometry_msgs.msg import Pose
from uuv_gazebo_ros_plugins_msgs.msg import FloatStamped
from vandy_bluerov.msg import HSDCommand
from sensor_msgs.msg import Image, CompressedImage
from scipy.signal import find_peaks 
from std_msgs.msg import Bool
from std_msgs.msg import Header
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
# from cv2_np_bridge import CvBridge
import time
import warnings
import copy
# from PIL import Image as PIL_Image
from std_msgs.msg import Float32MultiArray

# ALC TOOLCHAIN - VAE AM
#removing any gpu usage
import rospkg
import os
#os.environ["CUDA_VISIBLE_DEVICES"]=""

#forcing to use the alc_utils in the cpu_test folder
import sys
from datetime import datetime 

import alc_utils.assurance_monitor
import pickle
from torchvision import transforms
import torch
import threading

class LEC2LiteAM(object):
    def __init__(self):
        self.namespace = rospy.get_namespace().replace('/', '')
        self.side = rospy.get_param('~side', 'r')
        self.am_type = rospy.get_param('~am_type', 'vae')
        assert self.am_type in ['vae', 'svdd'], "Not supported AM type"
        self.cvbridge = CvBridge()

        self.am_lec2lite_pub = rospy.Publisher(
            '/vu_sss/am_' + self.am_type + '_lec2lite_' + self.side, 
            Float32MultiArray, 
            queue_size=1) 

        ###### ALC VAE AM
        #use the am in the current folder
        rospack = rospkg.RosPack()
        am_path = os.environ["ALC_WORKING_DIR"] + rospy.get_param('~path', "/jupyter/admin_BlueROV/LEC2Lite_AM/vae/SLModel")
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

        self.sss_waterfall_sub = rospy.Subscriber(
            '/vu_sss/waterfall_' + self.side, Image, self.callback_am, queue_size=1)

    def get_am(self, image):
        # image = transforms.Grayscale()(image)
        # image = self.transform(image)
        # Normalizing
        image = torch.div(image, 255)        
        # Input reshape
        x = image.reshape(1,1,16,180)
        return self.am.evaluate(x, None)

    def am_thread(self, msg):
        image = self.cvbridge.imgmsg_to_cv2(msg, "mono8")
        t = transforms.functional.to_tensor(image)
        # t = transforms.functional.to_tensor(self.cvbridge.imgmsg_to_cv2(msg, channels=1))
        # am_output = self.get_am(t.permute(2,0,1)) 
        
        start_time = datetime.now() 
        am_output = self.get_am(t) 
        time_elapsed = datetime.now() - start_time 
        
        am_output = am_output[0]
        # am_output[-2] = np.log(am_output[-2]) # log(m)   
        rospy.loginfo(" >> LEC2 {}-{}-AM \tlog(m): {}, det: {} Time elapsed (hh:mm:ss.ms) {}".format(self.am_type, self.side, am_output[-2], am_output[-1], time_elapsed))
        am_msg = Float32MultiArray()
        am_msg.data = am_output
        self.am_lec2lite_pub.publish(am_msg)

    def callback_am(self, msg):
        t = threading.Thread(target=self.am_thread, args=(msg,))
        t.start()

if __name__=='__main__':
    rospy.init_node('lec2lite_am', log_level=rospy.INFO)
    try:
        node = LEC2LiteAM()
        rospy.spin()
    except rospy.ROSInterruptException:
        print('caught exception')
