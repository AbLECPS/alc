#!/usr/bin/env python


'''
    This script will simulate a forward looking imaging sonar using a gazebo lidar
'''

# Following line changes the order in which Python checks for packages
import os
import sys
import rospy
import numpy as np
import math
import rospkg
import tf.transformations as trans
import scipy.misc
import cv2
import copy
from sensor_msgs.msg import LaserScan, Image
from std_msgs.msg import Float32MultiArray
from nav_msgs.msg import Odometry
from collections import deque
from cv_bridge import CvBridge
from scipy import interpolate
from scipy.spatial.transform import Rotation as R
import torch

class GT():
    SEAFLOOR =  [0,   255, 0]
    PIPE =      [255, 0,   0]
    UNKNOWN =   [0,   0,   0]
    DEBRIS =    [0,   0, 255]

class ForwardImagingSonar(object):
    def __init__(self):
        np.set_printoptions(suppress=True)
        # Get RosParams
        self.namespace = rospy.get_namespace().replace('/', '')
        self.z_scale = int(rospy.get_param('~z_scale', 2))
        self.beam_width = float(rospy.get_param('~beam_width', 1.5708))
        self.sonar_noise = float(rospy.get_param('~sonar_noise', 25))
        self.fis_bins = int(rospy.get_param('~fis_bins', 360))
        # Compared to 15m altitude -> Ideal scan altitude
        self.nominal_alt = int(rospy.get_param('~nominal_alt', 15))
        self.laser_topic = rospy.get_param('~laser_topic', "")
        print("[ VU FIS ] laser topic: ", self.laser_topic)
        self.uuv_rpy =  None
        self.uuv_position = [0,0,0]
        self.delta_source = None
        self.d_trans = [0,0,0]
        self.d_rot = [0,0,0]
        self.fis_scan = []
        self.fis_gt_scan = []
        self.pcl_size = 250
        self.pcl_dq = deque(maxlen=self.pcl_size)
        rp = rospkg.RosPack()
        filename = rp.get_path("vandy_bluerov") + "/nodes/sonars/fis_mask.png"
        self.sonar_mask = cv2.imread(filename, 0) 
        

        # Initialize subscribers/publishers
        self.fis_scan_pub = rospy.Publisher(
            '/vu_fis/scan', Image, queue_size=1)
        
        self.fis_scan_gt_pub = rospy.Publisher(
            '/vu_fis/scan_gt', Image, queue_size=1)

        self.cvbridge = CvBridge()
        self.odometry_sub = rospy.Subscriber(
             '/uuv0/pose_gt_noisy_ned', Odometry, self.callback_odometry, queue_size=1) 
        
        if self.laser_topic == '/scan':
            self.sub = rospy.Subscriber(
                "/scan", LaserScan, self.rplidar_callback, queue_size=1)     
        else:
            self.sub = rospy.Subscriber(
                "vu_fis", LaserScan, self.laser_callback, queue_size=1)
        
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            if len(self.fis_scan) > 0:
                # Create scan and gt image message and publish them
                msg, msg_gt = self.create_cvimage(self.fis_scan, sonar_noise=self.sonar_noise)
                msg.header.frame_id = "fis"
                self.fis_scan_pub.publish(msg)

                msg_gt.header.frame_id = "fis"
                self.fis_scan_gt_pub.publish(msg_gt)
                
                rate.sleep()

    def rplidar_callback(self, msg):
        '''
        This method handles input from real HW (RPLidar) to create FIS
        '''
        # RpLidar defaults:
        # current scan mode: Sensitivity, sample rate: 8 Khz, max_distance: 12.0 m, scan frequency:10.0 Hz, 

        # Transform rays to height scan line
        ranges = np.array(msg.ranges)     
        # print(np.ma.masked_invalid(abs(ranges)).mean())   
        ranges *= 40 # self.nominal_alt / np.ma.masked_invalid(abs(ranges)).mean()
        scan = self.transform_scan(ranges[405:765], self.fis_bins)
        self.transform_pcl()
        self.pcl_dq.append(scan)
        self.fis_scan = self.pcl_to_image(self.pcl_dq)
        self.fis_gt_scan = self.pcl_to_image(self.pcl_dq, gt=True)
    
    def laser_callback(self, msg):
        '''
        This method handles input from Gazebo (simulated FIS)
        '''
        # Transform rays to height scan line
        # print("laser_callback")
        scan = self.transform_scan(np.array(msg.ranges), self.fis_bins)
        
        self.transform_pcl()
        self.pcl_dq.append(scan)
        self.fis_scan = self.pcl_to_image(self.pcl_dq)
        self.fis_gt_scan = self.pcl_to_image(self.pcl_dq, gt=True)
        
    def transform_pcl(self):
        if self.delta_source is not None:
            self.d_trans = np.array(self.uuv_position) - np.array(self.delta_source[0])
            self.d_rot = np.array(self.uuv_rpy) - np.array(self.delta_source[1])
            # Ignore d_roll
            # self.d_rot[0] = 0
            # Ignore d_pitch
            # self.d_rot[1] = 0
            # Calculate fwd movement
            d_fwd_trans = math.sqrt(2*self.d_trans[0]**2)
        self.delta_source = [self.uuv_position, self.uuv_rpy]
        
            
        if len(self.pcl_dq) > 0:
            for i in range(len(self.pcl_dq)):
                # apply translation
                self.pcl_dq[i][:] -= [0, d_fwd_trans, -self.d_trans[2]]
                # apply Yaw rotation
                r = R.from_euler('zxy', self.d_rot, degrees=False)
                self.pcl_dq[i][:] = r.apply(self.pcl_dq[i][:]) 
    
    def pcl_to_image(self, pcl, img_size=(240, 240), gt=False):
        img=np.full(img_size, -np.inf)
        scale = 7
        offset = 80
        for points in pcl:
            for point in points:
                # print(point)
                coord = tuple((
                    int(img_size[0] - point[1]*scale - offset), 
                    int(point[0]*scale + img_size[1]//2) 
                    )) 
                if self.check_valid_coord(coord, img_size):
                    if not gt:
                        img[coord] = 255 - point[2]*10
                    else:
                        img[coord] = point[2]
        return img
    
    
    def check_valid_coord(self, coord, img_size):
        if 0 <= coord[0] < img_size[0] and 0 <= coord[1] < img_size[1]:
            return True
        return False

    def create_cvimage(self, img, sonar_noise=25):
        # ratio = 1/((np.nanmax(img[img != np.inf]) - np.nanmin(img[img != -np.inf]))/255)
        # img -= np.nanmin(img[img != -np.inf])
        # print("*** "+str(ratio)+" *** "+str(np.nanmin(img[img != -np.inf])))

        ratio = 7
        img -= 400
        img *= ratio * 0.75
        img[img != -np.inf] = 32+255-img[img != -np.inf]

        # Closing
        kernel = np.ones((5,1),np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

        #Passing for gt
        raw_scan = copy.deepcopy(img)
        # raw_scan[img == -np.inf] = np.nanmin(img[img != -np.inf])
        gt_img = self.create_gt_cvimage(raw_scan)

        # adding noise
        noise = np.random.normal(0, sonar_noise, np.shape(img))
        img[img>0] += noise[img>0]

        img[img>255]=255
        img[img<0]=0

        # Disortion for making radial blur
        pts1 = np.float32([[50,0],[189,0],[0,239],[239,239]])
        pts2 = np.float32([[0,0],[239,0],[0,239],[239,239]])
        M = cv2.getPerspectiveTransform(pts2,pts1)
        img = cv2.warpPerspective(img,M,(240,240))

        # sonar noise, motion/radial blur
        size = 9
        kernel_motion_blur = np.zeros((size, size))
        kernel_motion_blur[:, int((size-1)/2)] = np.ones(size)
        kernel_motion_blur = kernel_motion_blur / size
        
        # applying the kernel to the input image
        img = cv2.filter2D(img, -1, kernel_motion_blur)
        
        # reverting disortion
        M = cv2.getPerspectiveTransform(pts1,pts2)
        img = cv2.warpPerspective(img,M,(240,240))
        
        img[img>255]=255
        img[img<0]=0

        # apply masks
        img = cv2.bitwise_and(img, img, mask=self.sonar_mask) 
        gt_img = cv2.bitwise_and(gt_img, gt_img, mask=self.sonar_mask) 

        return  self.cvbridge.cv2_to_imgmsg(img.astype(np.uint8), encoding="mono8"), self.cvbridge.cv2_to_imgmsg(gt_img.astype(np.uint8), encoding="rgb8")
        

    def create_gt_cvimage(self, img):
        # Color GT Scan            
        img[img>255]=255
        img[img<0]=0
        mean_val = np.nanmean(img[img != -np.inf])//1

        # Calculated GT
        gt_img = np.zeros((np.shape(img)[0], np.shape(img)[1], 3))
        threshold_g = cv2.adaptiveThreshold(cv2.GaussianBlur(img.astype(np.uint8),(5,5),0),255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,55,0)

        # Define lower/upper value
        lower = mean_val - 10
        upper = 255

        # Check the region of the image actually with a color in the range defined below
        # inRange returns a matrix in black and white
        # threshold_g = cv2.inRange(img, lower, upper)

        # img = np.where(threshold_g <= 0, -3000, img) #  Unknown BLACK
        img_t = copy.deepcopy(img)
        # img[img!=] = -3000
        try:
            img_t[threshold_g == 0] = mean_val
        except ValueError:  
            pass
        blur = cv2.GaussianBlur(img_t.astype(np.uint8),(5,5),0)
        _,threshold = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        img = np.where(threshold > 0, -1000, img)
        img = np.where(threshold <= 0, 0, img)
        img[threshold_g == 0] = -3000
        

        # img = np.where(threshold > 0, -2000, img) # Debris/Unknown BLUE
        # img = np.where(np.logical_and(img > self.nominal_alt , img <= self.nominal_alt+50), -1000, img) # Pipeline RED
        # img = np.where(img > mean_val-100, 0, img) # Seabed GREEN

        gt_img[np.array(img) == 0] = GT.SEAFLOOR
        gt_img[np.array(img) == -1000] = GT.PIPE
        gt_img[np.array(img) == -2000] = GT.PIPE #GT.DEBRIS
        gt_img[np.array(img) <  -2000] = GT.UNKNOWN

        # Denoise/Close
        kernel = np.ones((5,1),np.uint8)
        gt_img = cv2.morphologyEx(gt_img, cv2.MORPH_CLOSE, kernel)

        # Return RGB image
        return gt_img

    def transform_scan(self, rays, fis_width):        
        # using GT depth
        # nominal_alt = 60 - self.uuv_position[2] # Nominal sea depth - GT depth
        nominal_alt = self.nominal_alt     
        vertical_offset_angle = -self.beam_width/2
        ray_step_angle = self.beam_width / fis_width
        pcl = []   

        for step in range(fis_width):   
            # Project rays to the reference flat seafloor
            if self.uuv_rpy is not None:
                roll_compensation = self.uuv_rpy[0]
                pitch_compensation = self.uuv_rpy[1] 
            else:
                roll_compensation = 0
                pitch_compensation = 0
                       
            scan_alpha = ((step + 0.5) * ray_step_angle + vertical_offset_angle)            
            sonar_pitch = math.radians(-45)
            
            # information from movement
            delta_rot = [0,0,0]
            delta_trans = [0,0,0]

            # [y p r]
            rot = [ 
                scan_alpha,
                sonar_pitch + pitch_compensation,
                roll_compensation
            ]

            echo = [0, rays[step], 0]

            if np.isfinite(rays[step]):
                r = R.from_euler('zxy', rot, degrees=False)
                point = r.apply(echo)
                pcl.append(point)
                # print(str(math.degrees(scan_alpha)) + " " + str(point))
        return np.array(pcl)

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
        uuv_rpy = trans.euler_from_quaternion(q, axes='sxyz')
        self.uuv_rpy = uuv_rpy 
        
    def vector_to_np(self, v):
        return np.array([v.x, v.y, v.z])
    
    def quaternion_to_np(self, q):
        return np.array([q.x, q.y, q.z, q.w])

    

if __name__ == "__main__":
    rospy.init_node("FIS_Waterfall")
    fis = ForwardImagingSonar()
    rospy.spin()
