#!/usr/bin/env python
import os
import rospy
import numpy as np
import math
import tf.transformations as trans
import csv

from geometry_msgs.msg import Point, Pose
from nav_msgs.msg import Odometry

class FDRReceiver(object):
    '''
    Acoustic sonar class for FDR ping detection (Cp3 debris detection)
    '''
    def __init__(self):
        np.random.seed(rospy.get_param('~random_seed'))
        self.log_filename = rospy.get_param('~log_filename', 'fdr_rx_map.csv')
        self.fdr_pub = rospy.Subscriber(
            '/fdr0/ping', Point, self.callback_fdr, queue_size=1)  
        self.uuv_position = [0,0,0]

        self.odometry_sub = rospy.Subscriber(
             '/uuv0/pose_gt_noisy_ned', Odometry, self.callback_odometry, queue_size=1)            

        self.noise_threshold = 1

        self.write_to_file(['x','y','signal_strength'], True)


    def callback_odometry(self, msg):
        self.uuv_position = self.get_position(msg)
       
    def callback_fdr(self, msg):
        signal = self.calculate_fdr_signal(
            self.spatial_distance(self.uuv_position, np.array([msg.y, msg.x, -msg.z])),
            self.noise_threshold,
            0.95,
            1)
        new_map_line = [self.uuv_position[1], self.uuv_position[0], signal]
        self.write_to_file(new_map_line)
        # print(signal)

    def write_to_file(self, data, new_file=False):
        if os.path.isdir(results_dir):
            rospy.loginfo("[FDR_RX]\033[1;32m writing file: " + os.path.join(results_dir, self.log_filename)+ " \033[0m")
            if not new_file:
                mode = 'a'
            else:
                mode = 'w'
            with open(os.path.join(results_dir, self.log_filename), mode) as fd:
                writer = csv.writer(fd)
                writer.writerow(data)
                fd.close()
        else:
            rospy.logwarn("[FDR_RX] log file path error")
   
    def calculate_fdr_signal(self, distance, noise_threshold, noise_low, noise_high):
        '''
        inverse square law
        FDR: 160dB @ 37.5kHz -> Can be detected from 1-2km in normal conditions
        '''
        distance = min(distance, 2000)
        # Generated with Matlab cftool
        # General model Rat21:
        # Coefficients (with 95% confidence bounds):
        p1 =     0.02082 # (-0.03527, 0.0769)
        p2 =      -75.45 # (-223, 72.09)
        p3 =   6.982e+04 # (-1.901e+04, 1.586e+05)
        q1 =       436.3 # (-121.9, 994.6)   
        fx = (p1*distance**2 + p2*distance + p3) / (distance + q1) # dB
        return fx * np.random.uniform(low=noise_low, high=noise_high) # signal strength with noise

    def get_position(self, odom):
        return self.vector_to_np(odom.pose.pose.position)

    
    def planar_distance(self, base, pos):
        return np.sqrt((base[0] - pos[0])**2 +
                       (base[1] - pos[1])**2)

    def vector_to_np(self, v):
        return np.array([v.x, v.y, v.z])

    
    def spatial_distance(self, base, pos):
        return np.sqrt((base[0] - pos[0])**2 +
                       (base[1] - pos[1])**2 +
                       (base[2] - pos[2])**2)

if __name__=='__main__':
    print('Starting FlightDataRecorder: FDRReceiver')
    rospy.init_node('fdr_receiver', log_level=rospy.INFO)
    results_dir = rospy.get_param("~results_directory")
    try:
        node = FDRReceiver()
        rospy.spin()
    except rospy.ROSInterruptException:
        print('caught exception')  
