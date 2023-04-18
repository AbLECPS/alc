#!/usr/bin/env python


'''
    This script will simulate a side scan sonar using a gazebo lidar
'''

# Following line changes the order in which Python checks for packages
import os
import sys
import rospy
import numpy as np
import math
import tf.transformations as trans
import scipy.misc
import cv2
from sensor_msgs.msg import LaserScan, Image
from std_msgs.msg import Float32MultiArray
from nav_msgs.msg import Odometry
from collections import deque
from cv_bridge import CvBridge
from scipy import interpolate

class GT():
    SEAFLOOR =  [0,   255, 0]
    PIPE =      [0,   0, 255]
    UNKNOWN =   [0,   0,   0]
    DEBRIS =    [255, 0,   0]

class SideScanSonar(object):
    def __init__(self):
        np.set_printoptions(suppress=True)
        # Get RosParams
        self.namespace = rospy.get_namespace().replace('/', '')
        self.z_scale = int(rospy.get_param('~z_scale', 16))
        self.y_scale = int(rospy.get_param('~y_scale', 8))
        self.sss_bins = int(rospy.get_param('~num_range_bins', 2048))
        self.sss_lines = int(rospy.get_param('~sss_lines', 256))
        self.beam_width = float(rospy.get_param('~beam_width', 1.3963))
        self.sonar_noise = float(rospy.get_param('~sonar_noise', 0.5))
        # Compared to 15m altitude -> Ideal scan altitude
        self.nominal_alt = int(rospy.get_param('~nominal_alt', 15))
        self.sss_waterfall = deque(maxlen=self.sss_lines)
        self.sss_waterfall_gt = deque(maxlen=self.sss_lines)
        self.orientation_lag = int(rospy.get_param('~orientation_lag', 1))
        self.laser_topic = rospy.get_param('~laser_topic', "")
        print("[ VU SSS ] laser topic: ", self.laser_topic)
        self.uuv_rpy =  [0,0,0]
        self.uuv_position = [0,0,0]

        # Initialize subscribers/publishers
        self.sss_waterfall_pub = rospy.Publisher(
            '/vu_sss/waterfall', Image, queue_size=1)
        
        self.sss_waterfall_gt_pub = rospy.Publisher(
            '/vu_sss/waterfall_gt', Image, queue_size=1)

        self.sss_waterfall_l_pub = rospy.Publisher(
            '/vu_sss/waterfall_l', Image, queue_size=1)
        self.sss_waterfall_r_pub = rospy.Publisher(
            '/vu_sss/waterfall_r', Image, queue_size=1)
        
        self.sss_waterfall_gt_l_pub = rospy.Publisher(
            '/vu_sss/waterfall_gt_l', Image, queue_size=1)    
        self.sss_waterfall_gt_r_pub = rospy.Publisher(
            '/vu_sss/waterfall_gt_r', Image, queue_size=1)    

        self.cvbridge = CvBridge()
        self.odometry_sub = rospy.Subscriber(
             '/uuv0/pose_gt_noisy_ned', Odometry, self.callback_odometry, queue_size=1) 
        
        if self.laser_topic == '/scan':
            self.sub = rospy.Subscriber(
                "/scan", LaserScan, self.rplidar_callback, queue_size=1)
                
        else:
            self.sub = rospy.Subscriber(
                "vu_sss", LaserScan, self.laser_callback, queue_size=1)
        
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            if len(self.sss_waterfall) == self.sss_lines:
                # Split Image:
                msg=self.create_cvimage(self.split_image(self.sss_waterfall, 0), self.z_scale)
                msg.header.frame_id = "sss_left"
                self.sss_waterfall_l_pub.publish(msg)
                msg = self.create_cvimage(self.split_image(self.sss_waterfall_gt, 0), self.z_scale, True)
                msg.header.frame_id = "sss_left"
                self.sss_waterfall_gt_l_pub.publish(msg)

                msg = self.create_cvimage(self.split_image(self.sss_waterfall, 1), self.z_scale)
                msg.header.frame_id = "sss_right"
                self.sss_waterfall_r_pub.publish(msg)
                msg = self.create_cvimage(self.split_image(self.sss_waterfall_gt, 1), self.z_scale, True)
                msg.header.frame_id = "sss_right"
                self.sss_waterfall_gt_r_pub.publish(msg)
                
                rate.sleep()

    def rplidar_callback(self, msg):
        '''
        This method handles input from real HW (RPLidar) to create SSS
        '''
        # RpLidar defaults:
        # current scan mode: Sensitivity, sample rate: 8 Khz, max_distance: 12.0 m, scan frequency:10.0 Hz, 

        # Transform rays to height scan line
        ranges = np.array(msg.ranges)     
        # print(np.ma.masked_invalid(abs(ranges)).mean())   
        ranges *= 40 # self.nominal_alt / np.ma.masked_invalid(abs(ranges)).mean()
        scan = self.transform_scan(ranges[405:765], self.sss_bins, self.y_scale)
        
        # Create GT
        scan_gt = self.create_ground_truth(scan)
        # Adding noise        
        scan += np.random.normal(0, self.sonar_noise, len(scan))

        # Append lines to waterfall images
        self.sss_waterfall.append(scan)
        self.sss_waterfall_gt.append(scan_gt)
    
        # if len(self.sss_waterfall) == self.sss_lines:
        #     # Split Image:
        #     msg=self.create_cvimage(self.split_image(self.sss_waterfall, 0), self.z_scale)
        #     msg.header.frame_id = "sss_left"
        #     self.sss_waterfall_l_pub.publish(msg)
        #     msg = self.create_cvimage(self.split_image(self.sss_waterfall_gt, 0), self.z_scale, True)
        #     msg.header.frame_id = "sss_left"
        #     self.sss_waterfall_gt_l_pub.publish(msg)

        #     msg = self.create_cvimage(self.split_image(self.sss_waterfall, 1), self.z_scale)
        #     msg.header.frame_id = "sss_right"
        #     self.sss_waterfall_r_pub.publish(msg)
        #     msg = self.create_cvimage(self.split_image(self.sss_waterfall_gt, 1), self.z_scale, True)
        #     msg.header.frame_id = "sss_right"
        #     self.sss_waterfall_gt_r_pub.publish(msg)

        #     # # Combined Image:          
        #     # self.sss_waterfall_pub.publish(self.create_cvimage(self.sss_waterfall, self.z_scale))
        #     # self.sss_waterfall_gt_pub.publish(self.create_cvimage(self.sss_waterfall_gt, self.z_scale, True))
        #     # # print('SSS Waterfall update')
    
    def laser_callback(self, msg):
        '''
        This method handles input from Gazebo (simulated SSS)
        '''
        # Transform rays to height scan line
        scan = self.transform_scan(np.array(msg.ranges), self.sss_bins, self.y_scale)
        # Matching sim to hw shift:
        scan -= np.nanmedian(scan)        
        scan *= 1.15
        # scan += 0.25

        # Create GT
        scan_gt = self.create_ground_truth(scan)
        # Adding noise        
        scan += np.random.normal(0, self.sonar_noise, len(scan))
        
        # Append lines to waterfall images
        self.sss_waterfall.append(scan)
        self.sss_waterfall_gt.append(scan_gt)
    
        # if len(self.sss_waterfall) == self.sss_lines:
        #     # Split Image:
        #     self.sss_waterfall_l_pub.publish(self.create_cvimage(self.split_image(self.sss_waterfall, 0), self.z_scale))
        #     self.sss_waterfall_gt_l_pub.publish(self.create_cvimage(self.split_image(self.sss_waterfall_gt, 0), self.z_scale, True))

        #     self.sss_waterfall_r_pub.publish(self.create_cvimage(self.split_image(self.sss_waterfall, 1), self.z_scale))
        #     self.sss_waterfall_gt_r_pub.publish(self.create_cvimage(self.split_image(self.sss_waterfall_gt, 1), self.z_scale, True))

    
    def split_image(self, image, id):
        np_img = np.array(image)
        if id > 0:
            # left
            return np_img[0:np_img.shape[0], 0:np_img.shape[1]//2]
        else:
            # right
            return np_img[0:np_img.shape[0], np_img.shape[1]//2:np_img.shape[1]]

    def create_cvimage(self, waterfall, z_scale, gt=False):
        if not gt:
            # Grayscale Side Scan
            img = 64 + np.array(waterfall)*z_scale
            img[img>255]=255
            img[img<0]=0
            img = np.flip(img, 0)
            img = np.flip(img, 1)
            # print(np.nanmean(np.nonzero(img)))
            img = self.cvbridge.cv2_to_imgmsg(img.astype(np.uint8), encoding="mono8")
        else:
            # Color GT Scan            
            img = np.zeros((np.shape(waterfall)[0], np.shape(waterfall)[1], 3))
            img[np.array(waterfall) == 0] = GT.SEAFLOOR
            img[np.array(waterfall) == 10] = GT.PIPE
            img[np.array(waterfall) == 20] = GT.PIPE
            img[np.array(waterfall) >  20] = GT.UNKNOWN
            # Flip x,y axis
            img = np.flip(img, 0)
            img = np.flip(img, 1)
            # Denoise
            kernel = np.ones((3,1),np.uint8)
            img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
            img = self.cvbridge.cv2_to_imgmsg(img.astype(np.uint8), encoding="bgr8")             
        return img

    def create_ground_truth(self, scan):    
        # scan = np.array(scan)    
        scan = np.where(scan > 1.5, 20, scan) # Debris/Unknown BLUE
        scan = np.where(scan == -np.inf, 30, scan) # Unknown / shadow BLACK
        scan = np.where(np.logical_and(scan >= 0.5, scan <= 1.5), 10, scan) # Pipeline RED
        scan = np.where(scan < 0.5, 0, scan) # Seabed GREEN
        return scan

    def transform_scan(self, rays, sss_width, y_scale = 12):        
        # using GT depth
        # nominal_alt = 60 - self.uuv_position[2] # Nominal sea depth - GT depth
        nominal_alt = self.nominal_alt     
        beam_width = self.beam_width * 2 # from gazebo.xacro, 2 sides
        vertical_offset_angle = -self.beam_width #(np.pi/2 - beam_width) / 2
        ray_step_angle = beam_width / sss_width
        scan = np.full(sss_width, -np.inf)       

        for step in range(sss_width):                        
            # Create nadir gap
            if abs(step - sss_width // 2) < 50:
                rays[step] = -np.inf
            
            # Project rays to the reference flat seafloor
            if len(self.uuv_rpy) > 0:
                roll_compensation = -self.uuv_rpy[0] # oldest in the deque
                pitch_compensation = -self.uuv_rpy[1] # oldest in the deque
            else:
                roll_compensation = 0
                pitch_compensation = 0
            alpha = np.pi/2 + roll_compensation - ((step + 0.5) * ray_step_angle + vertical_offset_angle)            
            # alpha = np.pi/2 - (step * ray_step_angle + vertical_offset_angle)            
            h1 = abs(math.sin(alpha) * rays[step]) * math.cos(pitch_compensation)# Echo altitude
            h2 = nominal_alt           
            d1 = math.cos(alpha) * rays[step] + sss_width//2
            d2 = nominal_alt / math.tan(alpha) + sss_width//2
            echo_alt = h2 - h1


            # Fill the shadows behind objects            
            # if np.isfinite(d2):
            #         indexes = self.get_idx(d2, sss_width, y_scale, step) # stretching scan
            #         for idx in indexes:
            #             if (0 < idx < sss_width) and echo_alt > 0:                  
            #                 scan[idx] = -np.inf 

            # Draw objects
            if np.isfinite(d1):
                    indexes = self.get_idx(d1, sss_width, y_scale, step) # stretching scan
                    for idx in indexes:
                        if (0 < idx < sss_width):
                            scan[idx] = echo_alt
                            # # If ray is outside (emulated) nadir gap 
                            # if (abs(sss_width//2 - step) < 200): #~10deg:
                            #     scan[idx] = -np.inf     
        return scan



    def get_idx(self, d, width, scale, step):
        indexes = []
        for i in range(max(int(abs(width//2 - d) * scale // 100 + 1), 1)):
        # for i in range(1):
            if d < width//2:
                idx = width//2 - (width//2 - d) * scale - i
            else:
                idx = (d - width//2) * scale + width//2 + i
            indexes.append(int(idx))
        return indexes
    
        # if d < width//2:
        #     idx = width//2 - (width//2 - d) * scale
        # else:
        #     idx = (d - width//2) * scale + width//2
        # indexes.append(int(idx))
        # return indexes

    def callback_odometry(self, msg):        
        pos = [msg.pose.pose.position.x,
               msg.pose.pose.position.y,
               msg.pose.pose.position.z]

        quat = [msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z,
                msg.pose.pose.orientation.w]

        # Calculate the position, position, and time of message
        p = self.vector_to_np(msg.pose.pose.position)
        self.uuv_position = p

        q = self.quaternion_to_np(msg.pose.pose.orientation)
        self.uuv_rpy = trans.euler_from_quaternion(q, axes='sxyz')
        
    def vector_to_np(self, v):
        return np.array([v.x, v.y, v.z])
    
    def quaternion_to_np(self, q):
        return np.array([q.x, q.y, q.z, q.w])

    

if __name__ == "__main__":
    rospy.init_node("SSS_Waterfall")
    sss = SideScanSonar()
    rospy.spin()
