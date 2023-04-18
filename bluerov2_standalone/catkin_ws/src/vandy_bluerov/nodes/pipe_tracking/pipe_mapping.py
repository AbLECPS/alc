#!/usr/bin/env python3

import rospy
import numpy as np
import math
import collections
import cv2
import warnings
import tf.transformations as trans

from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import Range
from geometry_msgs.msg import Pose
from uuv_gazebo_ros_plugins_msgs.msg import FloatStamped
from std_msgs.msg import Float32MultiArray
from vandy_bluerov.msg import HSDCommand
from sensor_msgs.msg import Image, CompressedImage
from scipy.signal import find_peaks 
from std_msgs.msg import Bool
from std_msgs.msg import Header
from nav_msgs.msg import Odometry
# from cv2_np_bridge import CvBridge
from cv_bridge import CvBridge

import cv2

import time
import copy
from PIL import Image as PIL_Image

class PipeMapping(object):
    def __init__(self):
        
        self.namespace = rospy.get_namespace().replace('/', '')
        rospy.loginfo('Initialize Pipe Mapping for <%s>' % self.namespace)
             
        self.cvbridge = CvBridge()

        self.sides = {
            "left" : 1,
            "right" : -1
        }

        self.size = 1000        
        pipeline_map_msg = OccupancyGrid()
        pipeline_map_msg.header.frame_id = self.namespace + '/pipeline_map'        
        pipeline_map_msg.info.resolution = 1
        pipeline_map_msg.info.width = self.size
        pipeline_map_msg.info.height = pipeline_map_msg.info.width
        pipeline_map_msg.info.origin.position.x = - self.size // 2 
        pipeline_map_msg.info.origin.position.y = pipeline_map_msg.info.origin.position.x 
        pipeline_map_msg.data = range(self.size*self.size)

        self.pipe_data = collections.deque(maxlen = 25)
        self.pipe_heading = 0
        self.hsd_pipe_mapping = 0
        self.pipe_distance = 0
        self.pipe_val = 50

        # Subscriber
        # Odom/Pose message
        self.odometry_sub = rospy.Subscriber(
             '/' + self.namespace + '/pose_gt_noisy_ned', Odometry, self.callback_odometry, queue_size=1) 
        # self.uuv_position = [0,0,0]
        self.uuv_position = collections.deque(maxlen = 50)
        self.position_lag = -1 #-50

        self.uuv_rpy = [0,0,0]

        # Altimeter
        self.range_sub = rospy.Subscriber(
            "/uuv0/altimeter_echosunder", Range, self.callback_range)

        # Subscribe to Left/Right VU SSS data 
        self.right_lec2_semseg_sub = rospy.Subscriber(
            '/vu_sss/lec2lite_r', Image, self.get_right_semseg)
        self.left_lec2_semseg_sub = rospy.Subscriber(
            '/vu_sss/lec2lite_l', Image, self.get_left_semseg)

        # Subscribe to Left/Right VU SSS data 
        self.right_gt_sub = rospy.Subscriber(
            '/vu_sss/waterfall_gt_r', Image, self.get_right_gt)
        self.left_gt_sub = rospy.Subscriber(
            '/vu_sss/waterfall_gt_l', Image, self.get_left_gt)


        # Evaluation
        # # Subscribe to Left/Right ground truth data
        # self.right_gt_sonar_sub = rospy.Subscriber(
        #     '/sss_sonar/left/data/ground_truth/compressed', CompressedImage, self.get_right_gt)
        # self.left_gt_sonar_sub = rospy.Subscriber(
        #     '/sss_sonar/left/data/ground_truth/compressed', CompressedImage, self.get_left_gt)

        # self.cm_use_lec2_sub = rospy.Subscriber(
        #     '/uuv0/cm_use_lec2', Bool, self.callback_cm_use_lec2)   
        # self.cm_use_lec2 = True

        self.bt_use_lec2_left_sub = rospy.Subscriber(
            "/bt/use_lec2/left", Bool, self.callback_bt_use_lec2_left) 
        self.bt_use_lec2_right_sub = rospy.Subscriber(
            "/bt/use_lec2/right", Bool, self.callback_bt_use_lec2_right) 
        self.bt_use_lec2 = {
            "left": True,
            "right": True
        }

        # Publisher
        self.lec2_accuracy_left_pub = rospy.Publisher(
            "/vu_sss/lec2_accuracy/left", Float32MultiArray, queue_size = 1)
        self.lec2_accuracy_right_pub = rospy.Publisher(
            "/vu_sss/lec2_accuracy/right", Float32MultiArray, queue_size = 1)

        # Pipe map
        self.pipeline_map_pub = rospy.Publisher(
            "/uuv0/pipeline_map", OccupancyGrid, queue_size = 1)
        
        # Pipe heading    
        self.pipeline_heading_pub = rospy.Publisher(
            "/uuv0/pipeline_heading_from_mapping", FloatStamped, queue_size = 1)
        pipeline_map_heading_msg = FloatStamped()
        
        # Pipe distance    
        self.pipeline_distance_pub = rospy.Publisher(
            "/uuv0/pipeline_distance_from_mapping", FloatStamped, queue_size = 1)
        pipeline_distance_msg = FloatStamped()

        # Pipe in view    
        self.pipeline_in_view_pub = rospy.Publisher(
            "/uuv0/pipeline_in_view", Header, queue_size = 1)
        self.pipeline_in_view_msg = Header()

        # Pipe in view GT
        self.pipeline_in_view_gt_pub = rospy.Publisher(
            "/uuv0/pipeline_in_view_gt", Header, queue_size = 1)
        

        # Pipe pos in SLS    
        self.pipeline_in_sls_pub = rospy.Publisher(
            "/uuv0/pipeline_in_sls", FloatStamped, queue_size = 1)
        self.pipeline_in_sls_msg = FloatStamped()
        
        # Pipe pos in SLS    
        self.pipeline_in_gt_pub = rospy.Publisher(
            "/uuv0/pipeline_in_gt", FloatStamped, queue_size = 1)        
        self.pipeline_in_gt_msg = FloatStamped()
      
        self.uuv_heading = 0
        self.uuv_altitude = 15 # By default
        self.uuv_pipeline_info = 0
        
        self.pipe_variances = {
            "gt_left": 0,
            "gt_right": 0,
            "lec_left": 0,
            "lec_right": 0
        }

        self.pipe_positions = {
            "gt_left": 0,
            "gt_right": 0,
            "lec_left": 0,
            "lec_right": 0
        }
       
        self.pipe_sls_pos = 0
        self.pipe_gt_pos = 0
        self.pipe_sls_stamp = rospy.Time.now() 
        self.pipe_gt_stamp = self.pipe_sls_stamp
        
        # Empty map
        self.map = np.ndarray((self.size, self.size), buffer=np.zeros((self.size, self.size), dtype=np.int), dtype=np.int)
        self.map.fill(0)

        # 1Hz loop
        rate = rospy.Rate(5)
        #
        # Pipe map update runs at 6-7Hz ...
        #
        while not rospy.is_shutdown():
            if rospy.Time(5) < rospy.Time.now():
                # Calculate pipeline heading
                # t = time.time()
                self.map_pipeline()
                # print(time.time() - t)
            # Publish map
            pipeline_map_msg.header.stamp = rospy.Time.now()
            pipeline_map_msg.data = self.map.flatten()
            self.pipeline_map_pub.publish(pipeline_map_msg)
            # Publish pipe info
            pipeline_map_heading_msg.header.stamp = rospy.Time.now()
            pipeline_map_heading_msg.data = self.pipe_heading
            self.pipeline_heading_pub.publish(pipeline_map_heading_msg)
            
            pipeline_distance_msg.header.stamp = rospy.Time.now()
            pipeline_distance_msg.data = self.pipe_distance
            self.pipeline_distance_pub.publish(pipeline_distance_msg)
            # print("pipeline published!")
            
            # rospy.logdebug('UUV: %0.1f deg, Pipe: %0.1f deg, H(SD): %0.1f deg ' %(self.uuv_heading, 90-self.pipe_heading, self.uuv_heading-(90-self.pipe_heading) ))         
            rate.sleep()


    
    def map_pipeline(self):
        t = time.time()
        nadir_gap_angle = math.radians(0.1)#1
        sls_beam_angle = math.radians(55)#70

        # If there is a valid & fresh reading
        # and CM approves LEC2 AM
        if len(self.uuv_position) > 0:
            # t = time.time()
            if (self.pipe_sls_pos != 0) and ((rospy.Time.now() - self.pipe_sls_stamp) < rospy.Duration(secs = 1.5)):

                # Calculate and publish LEC2 accuracy                
                self.calculate_lec2_accuracy()

                yaw = math.radians(90 - self.uuv_rpy[2])
        

                # p1 nadir gap/2 length on seafloor
                # p2 SLS beam length on seafloor
                p1 = math.tan(nadir_gap_angle) * self.uuv_altitude
                p2 = math.tan(nadir_gap_angle + sls_beam_angle) * self.uuv_altitude
                # Pipe distance from UUV in XY plane
                self.pipe_distance = (p2-p1) * self.pipe_sls_pos + p1
                # rospy.loginfo('                                         pipe distance: %d m' %self.pipe_distance)
                # Get R matrix on Z (yaw) axis
                R = np.array([[np.cos(yaw - np.sign(self.pipe_sls_pos) * 1.5708),   np.sin(yaw - np.sign(self.pipe_sls_pos) * 1.5708)],
                                        [-np.sin(yaw - np.sign(self.pipe_sls_pos) * 1.5708),  np.cos(yaw - np.sign(self.pipe_sls_pos) * 1.5708)]])
                # Get contact position
                pipe = np.dot(R, np.array([0, np.sign(self.pipe_sls_pos) * self.pipe_distance])) + np.array([self.uuv_position[self.position_lag][0] + self.size // 2, self.uuv_position[self.position_lag][1] + self.size // 2])
                # Draw pipe to map
                if 0 <= pipe[0] <= self.size and 0 <= pipe[1] <= self.size:  
                    self.map[int(pipe[0]), int(pipe[1])] = self.pipe_val      
                    self.pipe_data.append([int(pipe[0]), int(pipe[1])])
                    # Compute pipe heading as linear fitting
                    if len(self.pipe_data) > 5:
                        _pd = copy.deepcopy(self.pipe_data)
                        with warnings.catch_warnings():
                            warnings.filterwarnings('error')
                            try:
                                a,b = np.polyfit([column[0] for column in _pd],[column[1] for column in _pd],1) 
                                self.pipe_heading = self.get_heading_from_line(a,b)                
                            except np.RankWarning:
                                self.pipe_heading = 0.0
                    # Update pipeline in view message
                    self.pipeline_in_view_msg.stamp = rospy.Time.now()
                    self.pipeline_in_view_pub.publish(self.pipeline_in_view_msg)
                    self.pipeline_in_sls_msg.data = self.pipe_sls_pos
                    self.pipeline_in_sls_msg.header.stamp = self.pipeline_in_view_msg.stamp
                    self.pipeline_in_sls_pub.publish(self.pipeline_in_sls_msg)
                    # Publish GT for evaluation
                    self.pipeline_in_gt_msg.data = self.pipe_gt_pos
                    self.pipeline_in_gt_msg.header.stamp = self.pipeline_in_view_msg.stamp
                    self.pipeline_in_gt_pub.publish(self.pipeline_in_gt_msg)
                else:
                    rospy.loginfo("pipe pos is invalid")
            # else:
            #     rospy.loginfo('Got invalid or old pipeline data')
        # print(" > %s s" %(time.time() - t))

    def calculate_lec2_accuracy(self):             
        '''
        Calculate and publish LEC2 metrics based on computed GT for pipe perception
        '''

        lec2_metrics = {
            'LEFT': {
                'IOU': 0,
                'TP': 0,
                'TN': 0,
                'FP': 0,
                'FN': 0,
                'ACC': 0,
                'VAR': 0
            },
            'RIGHT': {
                'IOU': 0,
                'TP': 0,
                'TN': 0,
                'FP': 0,
                'FN': 0,
                'ACC': 0,
                'VAR': 0
            }    
        }
        
        lec2_accuracy_msg = Float32MultiArray()

        if self.pipe_positions["gt_left"] is None and self.pipe_positions["lec_left"] is not None:
            lec2_metrics['LEFT']['FP'] = 1
        elif self.pipe_positions["gt_left"] is not None and self.pipe_positions["lec_left"] is None:
            lec2_metrics['LEFT']['FN'] = 1
        elif self.pipe_positions["gt_left"] is None and self.pipe_positions["lec_left"] is None:
            lec2_metrics['LEFT']['TN'] = 1
        else:
            lec2_metrics['LEFT']['ACC'] = 1 - abs(self.pipe_positions["lec_left"] - self.pipe_positions["gt_left"])   
            lec2_metrics['LEFT']['TP'] = 1
            lec2_metrics['LEFT']['VAR'] = abs(self.pipe_variances["gt_left"] - self.pipe_variances["lec_left"])
            # Missing IOU

        lec2_accuracy_msg.data = [
            lec2_metrics['LEFT']['IOU'],
            lec2_metrics['LEFT']['TP'],
            lec2_metrics['LEFT']['TN'],
            lec2_metrics['LEFT']['FP'],
            lec2_metrics['LEFT']['FN'],
            lec2_metrics['LEFT']['ACC'],
            lec2_metrics['LEFT']['VAR']
            ]
        self.lec2_accuracy_left_pub.publish(lec2_accuracy_msg)



        if self.pipe_positions["gt_right"] is None and self.pipe_positions["lec_right"] is not None:
            lec2_metrics['RIGHT']['FP'] = 1
        elif self.pipe_positions["gt_right"] is not None and self.pipe_positions["lec_right"] is None:
            lec2_metrics['RIGHT']['FN'] = 1
        elif self.pipe_positions["gt_right"] is None and self.pipe_positions["lec_right"] is None:
            lec2_metrics['RIGHT']['TN'] = 1
        else:
            lec2_metrics['RIGHT']['ACC'] = 1 - abs(self.pipe_positions["lec_right"] - self.pipe_positions["gt_right"])
            lec2_metrics['RIGHT']['TP'] = 1 
            lec2_metrics['RIGHT']['VAR'] = abs(self.pipe_variances["gt_right"] - self.pipe_variances["lec_right"])
            # Missing IOU


        lec2_accuracy_msg.data = [
            lec2_metrics['RIGHT']['IOU'],
            lec2_metrics['RIGHT']['TP'],
            lec2_metrics['RIGHT']['TN'],
            lec2_metrics['RIGHT']['FP'],
            lec2_metrics['RIGHT']['FN'],
            lec2_metrics['RIGHT']['ACC'],
            lec2_metrics['RIGHT']['VAR']
            ]
        self.lec2_accuracy_right_pub.publish(lec2_accuracy_msg)

    def get_heading_from_line(self, a, b):            
        try:
            heading = math.degrees( math.atan2(a*10+b , 10-(-b/a)) ) #RuntimeWarning: divide by zero encountered in double_scalars
        except:
            heading = 0.0
            
        if abs(heading) > 90:
            heading = (180 - abs(heading)) * -np.sign(heading)
        return heading

    def get_heading_diff(self, alpha, beta):
        diff = (alpha-beta + 180) % 360 - 180
        return diff

    def callback_range(self, msg):
        self.uuv_altitude = msg.range

    def multi_dilation(self, im, num):
        for i in range(num):
            im = dilation(im)
        return im
        
    def multi_erosion(self, im, num):
        for i in range(num):
            im = erosion(im)
        return im
    
    def get_pipe_pos_from_semseg(self, msg, side):
        # For same node prosessing of LEC output: 
        image_np = copy.deepcopy(self.cvbridge.imgmsg_to_cv2(msg, "bgr8"))
        # Selecting R channel    
        image_np = image_np[:,:,2] 
        image_np = np.where(image_np<100,0,image_np)
        
        # looking for peaks in the image:
        pos_array = []
        [height, width] = np.shape(image_np)
        for i in range(height):
            peaks, _ =  find_peaks(image_np[i,:], height=35, width=2)
            if np.isnan(peaks.all()):
                continue
            if np.isreal(peaks.all()):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    pos_array.append(np.nanmean(peaks) / width)
        if len(pos_array) > height/4:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                pos = np.nanmean(pos_array)
                var = np.nanvar(pos_array)
            # print(var)
        else: 
            return None

        # rospy.loginfo('%d: %0.2f' %(side,pos))

        if not np.isnan(pos):
            # pipe_sls_stamp = msg.header.stamp
            if side > 0: # left
                pipe_sls_pos =  (1 - pos) * (-1)
                #     image_np = np.fliplr(image_np)
            else:
                pipe_sls_pos =  pos
            # print(pipe_sls_pos)
            #return [msg.header.stamp, pipe_sls_pos]
            return [rospy.Time.now(), pipe_sls_pos, var]
        else:
            return None
            
    def get_right_semseg(self, msg):
        ret = self.get_pipe_pos_from_semseg(msg, self.sides["right"])
        self.pipe_positions["lec_right"] = None
        if ret is not None:
            # print("r %0.2f" %self.pipe_sls_pos)
            if self.bt_use_lec2["right"]:
                # If AM is good - BT controlled:
                [self.pipe_sls_stamp, self.pipe_sls_pos, self.pipe_variances["lec_right"] ] = ret
            # else:
            #     print(" ==== RIGHT LEC2 DISABLED ====")
            # for metrics only:
            self.pipe_positions["lec_right"] = ret[1]

    def get_left_semseg(self, msg):
        ret = self.get_pipe_pos_from_semseg(msg, self.sides["left"])
        self.pipe_positions["lec_left"] = None
        if ret is not None:
            # print("l %0.2f" %self.pipe_sls_pos)
            if self.bt_use_lec2["left"]:
                # If AM is good - BT controlled:
                [self.pipe_sls_stamp, self.pipe_sls_pos, self.pipe_variances["lec_left"] ] = ret
            else:
                print(" ==== LEFT LEC2 DISABLED ====")
            # for metrics only:
            self.pipe_positions["lec_left"] = ret[1]     

    def get_right_gt(self, msg):
        ret = self.get_pipe_pos_from_semseg(msg, self.sides["right"])
        if ret is not None:
            # print("r %0.2f" %self.pipe_sls_pos)
            [self.pipe_gt_stamp, self.pipe_gt_pos, self.pipe_variances["gt_right"] ] = ret
            self.pipe_positions["gt_right"] = ret[1]

    def get_left_gt(self, msg):
        ret = self.get_pipe_pos_from_semseg(msg, self.sides["left"])
        if ret is not None:
            # print("l %0.2f" %self.pipe_sls_pos)
            [self.pipe_gt_stamp, self.pipe_gt_pos, self.pipe_variances["gt_left"] ] = ret 
            self.pipe_positions["gt_left"] = ret[1]
            # self.pipe_positions_variance[] = ret[2]

    # def callback_cm_use_lec2(self, msg):   
    #     self.cm_use_lec2 = msg.data
    def callback_bt_use_lec2_left(self, msg):
        self.bt_use_lec2["left"] = msg.data

    def callback_bt_use_lec2_right(self, msg):
        self.bt_use_lec2["right"] = msg.data

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
        self.uuv_position.append(p)

        # vel = self.vector_to_mag(msg.twist.twist.linear)

        q = self.quaternion_to_np(msg.pose.pose.orientation)
        rpy = trans.euler_from_quaternion(q, axes='sxyz')
        
        self.uuv_rpy[0] = math.degrees(rpy[0])
        self.uuv_rpy[1] = math.degrees(rpy[1])
        self.uuv_rpy[2] = math.degrees(rpy[2])

        self.uuv_heading = math.degrees(rpy[2])    
    
    def vector_to_np(self, v):
        return np.array([v.x, v.y, v.z])
    
    def quaternion_to_np(self, q):
        return np.array([q.x, q.y, q.z, q.w])

if __name__=='__main__':
    print('Starting pipeline mapping')
    # rospy.init_node('pipeline_mapping', log_level=rospy.DEBUG)
    rospy.init_node('pipeline_mapping', log_level=rospy.INFO)
    try:
        node = PipeMapping()
        rospy.spin()
    except rospy.ROSInterruptException:
        print('caught exception')
