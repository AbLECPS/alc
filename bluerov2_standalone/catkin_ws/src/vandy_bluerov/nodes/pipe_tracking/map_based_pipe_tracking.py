#!/usr/bin/env python

import rospy
import numpy as np
import math
import tf
import collections
import tf.transformations as trans

from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import Range
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point
from uuv_gazebo_ros_plugins_msgs.msg import FloatStamped
from vandy_bluerov.msg import HSDCommand
from nav_msgs.msg import Odometry




class MapBasedPipeTracking(object):
    def __init__(self):
        self.namespace = rospy.get_namespace().replace('/', '')
        rospy.loginfo('Initialize Map based Pipe Tracking for <%s>' % self.namespace)
             
        tf_transform = tf.TransformListener()
      
        # Subscriber
        # Altimeter
        self.range_sub = rospy.Subscriber(
            "/uuv0/altimeter_echosunder", Range, self.callback_range)
  
        # Pipe heading    
        self.pipeline_heading_sub = rospy.Subscriber(
            "/uuv0/pipeline_heading_from_mapping", FloatStamped, self.callback_heading)
        
        # Pipe distance    
        self.pipeline_distance_sub = rospy.Subscriber(
            "/uuv0/pipeline_distance_from_mapping", FloatStamped, self.callback_distance)
        
        self.sub = rospy.Subscriber(
            "/uuv0/pose_gt_noisy_ned", Odometry, self.callback_pose)
        self.uuv_yaw = 0 #rad
        self.uuv_position = Point()

        # Pipe pos in SLS    
        self.pipeline_in_sls_sub = rospy.Subscriber(
            "/uuv0/pipeline_in_sls", FloatStamped, self.callback_pipeline_in_sls)
        self.pipeline_in_sls = 0.0
        self.K_p_pipe_in_sls = 50
        self.pipe_sls_stamp = rospy.Time.now()

        # HSD publisher
        self.hsd_pipeline_mapping_pub = rospy.Publisher(
            "/uuv0/hsd_pipeline_mapping", HSDCommand, queue_size = 1)
        self.init_speed = rospy.get_param('~init_speed', 2.0)
        self.init_depth = rospy.get_param('~init_depth', 45)
        
        hsd_pipeline_mapping_msg = HSDCommand()
                
        self.uuv_heading = 0
        self.uuv_altitude = 0
        self.pipe_heading = 0
        self.pipeline_distance = 0

        # 1Hz loop
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            # Get TF transformation between world and UUV
            try:
                (trans,rot)  = tf_transform.lookupTransform("/world", "/uuv0/base_link", rospy.Time(secs=0))			
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                continue       

            rpy = tf.transformations.euler_from_quaternion(rot)
            self.uuv_heading = math.degrees(rpy[2])

            # Get UUV and pipe heading difference as desired heading
            hsd_pipeline_mapping_msg.heading = self.uuv_heading-(90-self.pipe_heading)
            
            # Slight modification of heading (when close to parallel) to bring pipe back to center of scan
            if abs(hsd_pipeline_mapping_msg.heading) < 10 and ((rospy.Time.now() - self.pipe_sls_stamp) < rospy.Duration(secs = 1.5)):               
                _mod = (abs(self.pipeline_in_sls) - 0.5) * self.K_p_pipe_in_sls * np.sign(self.pipeline_distance)
                # print("sls% "+str(self.pipeline_in_sls)+" "+str(_mod))
                hsd_pipeline_mapping_msg.heading += _mod
                # rospy.loginfo('%0.2f %0.2f' %(abs(self.pipeline_in_sls),_mod))
            hsd_pipeline_mapping_msg.speed = self.init_speed
            hsd_pipeline_mapping_msg.depth = self.init_depth
            hsd_pipeline_mapping_msg.header.stamp = rospy.Time.now()
            
            self.hsd_pipeline_mapping_pub.publish(hsd_pipeline_mapping_msg)

            rospy.logdebug(hsd_pipeline_mapping_msg.heading)
            
            rate.sleep()

    def callback_range(self, msg):
        self.uuv_altitude = msg.range

    
    def callback_heading(self, msg):
        self.pipe_heading = msg.data

    
    def callback_distance(self, msg):
        self.pipeline_distance = msg.data
        # rospy.loginfo(self.pipeline_distance )

    def callback_pose(self, odom):
        # Convert Quaternion to rpy
        rpy = trans.euler_from_quaternion([odom.pose.pose.orientation.x,
                                     odom.pose.pose.orientation.y,
                                     odom.pose.pose.orientation.z,
                                     odom.pose.pose.orientation.w])        
        self.uuv_position = odom.pose.pose.position
        self.uuv_yaw = rpy[2]
        # rospy.loginfo('UUV: %0.2f, %0.2f, %0.2f' %(self.uuv_position.x,self.uuv_position.y, math.degrees(self.uuv_yaw)))
    def callback_pipeline_in_sls(self, msg):
        self.pipeline_in_sls = msg.data
        self.pipe_sls_stamp = msg.header.stamp

if __name__=='__main__':
    print('Starting map based pipeline tracking')
    # rospy.init_node('map_based_pipe_tracking', log_level=rospy.DEBUG)
    rospy.init_node('map_based_pipe_tracking', log_level=rospy.INFO)
    try:
        node = MapBasedPipeTracking()
        rospy.spin()
    except rospy.ROSInterruptException:
        print('caught exception')
