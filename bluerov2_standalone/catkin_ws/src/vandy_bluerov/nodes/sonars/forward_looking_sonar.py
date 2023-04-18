#!/usr/bin/env python


'''
    This script will simulate a forward looking sonar using a gazebo ultrasonic sensors
'''

# Following line changes the order in which Python checks for packages
import os
import sys
import rospy
import numpy as np
import math
import cv2
import serial

from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Range
from collections import deque
from cv_bridge import CvBridge
from scipy.ndimage import gaussian_filter1d
from message_filters import ApproximateTimeSynchronizer, Subscriber

class ForwardLookingSonar(object):
    def __init__(self):
        np.set_printoptions(suppress=True)
        # Get RosParams
        self.namespace = rospy.get_namespace().replace('/', '')
        self.sonar_noise = float(rospy.get_param('~sonar_noise', 0.01))
        self.fls_bins = int(rospy.get_param('~num_range_bins', 252))
        self.fls_range = int(rospy.get_param('~fls_range', 30))
        
        self.use_hw_fls = rospy.get_param('~use_hw_fls', False)
        if self.use_hw_fls:
            self.fls_lines = 25
        else:
            self.fls_lines = rospy.get_param('~fls_lines', 100)

        self.fls_waterfall = deque(maxlen=self.fls_lines)
        self.fls_waterfall_gt = deque(maxlen=self.fls_lines)

        topic = rospy.get_param('~topic', '/vu_fls/bins')

        self.cvbridge = CvBridge()

        self.fls_raw = []   

        self.baseline_noise = [
            226, 226, 229, 229, 229, 226, 224, 190, 165, 148, 137, 112,  81,  70,  33,  16,   8,   5,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0 
            ]

        # Initialize subscribers/publishers
        self.fls_waterfall_pub = rospy.Publisher(
            '/vu_fls/waterfall', Image, queue_size=1)

        self.fls_bins_pub = rospy.Publisher(
            topic, Float32MultiArray, queue_size=1)

        if not self.use_hw_fls:
            self.fls_waterfall_gt_pub = rospy.Publisher(
                '/vu_fls/waterfall_gt', Image, queue_size=1)
        
            self.fls_bins_gt_pub = rospy.Publisher(
                '/vu_fls/bins_gt', Float32MultiArray, queue_size=1)

            self.fls_gt_pub = rospy.Publisher(
                '/vu_fls/gt', Float32MultiArray, queue_size=1)           

            sonar_0_sub = Subscriber('/uuv0/fls_sonar_0', Range)
            sonar_1_sub = Subscriber('/uuv0/fls_sonar_1', Range)
            sonar_2_sub = Subscriber('/uuv0/fls_sonar_2', Range)
            sonar_3_sub = Subscriber('/uuv0/fls_sonar_3', Range)
            sonar_4_sub = Subscriber('/uuv0/fls_sonar_4', Range)
            sonar_5_sub = Subscriber('/uuv0/fls_sonar_5', Range)
            sonar_6_sub = Subscriber('/uuv0/fls_sonar_6', Range)
            sonar_7_sub = Subscriber('/uuv0/fls_sonar_7', Range)
            sonar_8_sub = Subscriber('/uuv0/fls_sonar_8', Range)
            approxTimeSync=ApproximateTimeSynchronizer([sonar_0_sub,
                                                        sonar_1_sub,
                                                        sonar_2_sub,
                                                        sonar_3_sub,
                                                        sonar_4_sub,
                                                        sonar_5_sub,
                                                        sonar_6_sub,
                                                        sonar_7_sub,
                                                        sonar_8_sub], queue_size=1, slop=0.1)

            approxTimeSync.registerCallback(self.sonar_callback)
        else:
            serial_port = rospy.get_param('~serial_port', '/dev/ttyACM0')
            serial_baud = int(rospy.get_param('~serial_baud', 921600))
            self.ser = serial.Serial(serial_port, serial_baud)
            msg = Float32MultiArray()
            while not rospy.is_shutdown():
                bins = self.ser.readline().decode("utf-8")[1:-3] 
                bins = np.array(bins.split(", "), dtype='i')
                bins -= np.min(bins)
                bins = np.multiply(bins, 2.8).astype(int) - self.baseline_noise # scaling to 0-255
                bins[bins < 0] = 0
                bins_msg = Float32MultiArray()
                bins_msg.data = bins
                # print(bins)
                self.fls_bins_pub.publish(bins_msg)
                rospy.loginfo("FLS bins published")
                
                self.fls_waterfall.append(bins)
                
                if len(self.fls_waterfall) == self.fls_lines:
                    self.fls_waterfall_pub.publish(self.create_waterfall_msg(
                        self.fls_waterfall, 
                        252, #fixed for HW
                        self.fls_lines
                        ))

    def sonar_callback(self,*args):
        fls_raw = []
        # Append callback vals to fls list
        for idx, msg in enumerate(args):                  
            fls_raw.append(msg.range)
        
        # Compute 9x9-9 averages at intersecting cones
        # for i in range(9):
        #     for j in range(9):
        #         if i != j:
        #             # if theres is a valid reading
        #             if fls_raw[i] < self.fls_range and fls_raw[j]  < self.fls_range:
        #                 fls_raw.append(np.mean([fls_raw[i], fls_raw[j]]))
        
        gt_msg = Float32MultiArray()
        gt_msg.data = fls_raw
        
        bins, bins_raw = self.process_fls_bins(fls_raw)
        self.fls_waterfall.append(bins)
        self.fls_waterfall_gt.append(np.where(bins_raw>0, 255, 0))
        
        # Publish RAW bins as sonar output
        bins_msg = Float32MultiArray()
        bins_msg.data = bins
        self.fls_bins_pub.publish(bins_msg)
        bins_msg.data = bins_raw
        self.fls_bins_gt_pub.publish(bins_msg)
        
        # Publish Waterfall image as human "readable" version of the bins
        if len(self.fls_waterfall) == self.fls_lines:
            self.fls_waterfall_pub.publish(self.create_waterfall_msg(
                self.fls_waterfall, 
                self.fls_bins,
                self.fls_lines
                ))

            self.fls_waterfall_gt_pub.publish((self.create_waterfall_msg(
                self.fls_waterfall_gt, 
                self.fls_bins,
                self.fls_lines
                )))
            
            self.fls_gt_pub.publish(gt_msg)

    def create_waterfall_msg(self, waterfall, bins, lines):
        fls_image = Image()
        fls_image.width = bins
        fls_image.height = lines
        fls_image.header.stamp = rospy.Time.now()
        img = np.array(np.array(waterfall, dtype=np.uint8))
        # Flip H and V
        # Top = newest, left = closest
        img = np.flip(img, 0)
        return self.cvbridge.cv2_to_imgmsg(img, encoding="passthrough")

    def process_fls_bins(self, fls_raw, increment=128):
        bins = np.zeros((self.fls_bins))        
        bins_raw = np.zeros((self.fls_bins))        
        for i in range(len(fls_raw)):
            # get echo distances
            echo = float(fls_raw[i]) + np.random.rand(1) * 10 * self.sonar_noise
            echo = min(self.fls_range, max(0, echo[0]))
            # transform distances to bins
            id = int((echo/(self.fls_range)) * self.fls_bins)
            # Echo
            bins[id-1] += increment        
            # Harmonic echo
            for harmonic in [0.25, 0.5, 2, 4]:
                harmonic_id = max(min(self.fls_bins-1, int((id-1)* harmonic)), 0)
                if 0 < harmonic_id < self. fls_bins:
                    # distance based echo strength
                    bins[harmonic_id] += increment * (1 - (id-1)/self.fls_bins) / 2
        # smooth values        
        bins_raw = bins
        bins = gaussian_filter1d(bins, 3)
        # add noise
        noise = np.random.normal(0, 255 * self.sonar_noise, self.fls_bins)
        mask = np.where(np.random.normal(0, 1, self.fls_bins) > 1-self.sonar_noise, 1, 0)
        bins += noise * mask
        # Maximize values
        bins[bins > 255] = 255
        bins[bins < 0] = 0
        return bins, bins_raw

if __name__ == "__main__":
    rospy.init_node("Forward_Looking_Sonar")
    sss = ForwardLookingSonar()
    rospy.spin()
