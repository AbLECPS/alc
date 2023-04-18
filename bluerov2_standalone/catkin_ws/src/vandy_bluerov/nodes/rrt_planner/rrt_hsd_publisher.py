#!/usr/bin/env python

# Copyright (c) 2016 The UUV Simulator Authors.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified for VU in 2020

# TODO remove unnecessary imports
import numpy as np
import math
from scipy.spatial.transform import Rotation
from multiprocessing import Process

from math import sqrt
import random
from quads import QuadTree, BoundingBox

import math
import rospy
# import heading
import math
from scipy.spatial.transform import Rotation

from uuv_control_msgs.srv import *
from std_msgs.msg import String, Time, Float64MultiArray, Bool, Int32
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseStamped, Point, PoseArray, Pose
from nav_msgs.msg import OccupancyGrid
from vandy_bluerov.msg import HSDCommand

# Gets the yaw Euler angle from a quaternion in the form [x, y, z, w]
# Returns the yaw from the conversion
def get_yaw_from_quaternion(Q):
    return Rotation.from_quat([Q[0], Q[1], Q[2], Q[3]]).as_euler('XYZ')[2]

# Gets the rotation matrix from a quaternion in the form [x, y, z, w]
# Returns the rotation matrix from the conversion
def get_rotation_matrix_from_quaternion(Q):
    return Rotation.from_quat([Q[0], Q[1], Q[2], Q[3]]).as_dcm()

# Normalizes a radians angle to be between [-PI, PI]
# Returns the normalized yaw
def normalize(theta):
    while theta > math.pi:
        theta -= 2*math.pi
    while theta < -math.pi:
        theta += 2*math.pi

    return theta

# Converts x, y, theta from NED to ENU coordinate system
# Returns x, y, and theta in the ENU system
def NED_to_ENU(x, y, theta):    
    return y, x, normalize(math.pi / 2 - theta)

# Converts x, y, theta from ENU to NED coordinate system
# Returns x, y, and theta in the NED system
def ENU_to_NED(x, y, theta):    
    return y, x, normalize(math.pi / 2 - theta)

# Class for our HSD command publisher for the RRTPlanner
class RRTHSDPublisher():
    def __init__(self):
        self.stationary_heading = 90.0 # Holds the heading for "not moving" (spinning in place) the robot

        # Subscriber to whether or not to publish an HSD and move the robot
        self.publish_rrt_hsd_subscriber = rospy.Subscriber(
            '/uuv0/publish_rrt_hsd', Bool, self.publish_rrt_hsd_callback, queue_size=1) 
        self.hsd_should_publish = False # Holds whether or not to publish a move command

        # Subscriber for the path that the RRTPlanner generates
        self.rrt_path_subscriber = rospy.Subscriber('/uuv0/rrt_path', Path, self.rrt_path_callback, queue_size = 1)
        self.path = Path() # Holds the Path from the RRTPlanner
        
        # Subscriber to the robot position in NED coordinates
        self.robot_position = rospy.Subscriber(
            '/uuv0/pose_gt_noisy_ned', Odometry, self.callback_robot_position, queue_size=1) 
        self.robot_position = None # Holds the most recent robot position in ENU coordinates
        
        # Publishes commands to the robot for movement
        #self.hsd_pub = rospy.Publisher( '/uuv0/hsd_to_waypoint_rrt', HSDCommand, queue_size=1)
        self.hsd_pub = rospy.Publisher( '/uuv0/hsd_to_waypoint_rrt', HSDCommand, queue_size=1)
        self.hsd_msg =  HSDCommand() # Holds the HSD command to publish to the robot
        self.hsd_msg.header.seq = 1

        # Loop every 1 second (we need to supply HSDs every 1 second to be consistent)
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            if self.hsd_should_publish:
                self.get_hsd_from_path() # If we should publish a robot command, get our HSD from the path
            else:
                # DONT MOVE AT ALL (flip between turning each way)
                self.hsd_msg.heading = self.stationary_heading 
                self.stationary_heading  *= -1
                self.hsd_msg.speed = 0.0
                self.hsd_msg.depth = 45.0
            
            # publish our hsd message
            self.hsd_msg.header.frame_id = ''
            self.hsd_msg.header.seq += 1
            self.hsd_msg.header.stamp.secs += 1
            
            self.hsd_pub.publish(self.hsd_msg)

            # sleep
            rate.sleep()

    # Get our HSD from the path provided (moves in a straightline towards the second node in the path)
    # TODO Make this better (move towards the line rather than the endpoint)
    def get_hsd_from_path(self):
        # if we have a path and we have where the robot is
        if len(self.path.poses) >= 2 and self.robot_position != None:
            target = (self.path.poses[1].pose.position.x, self.path.poses[1].pose.position.y, get_yaw_from_quaternion([self.path.poses[1].pose.orientation.x, self.path.poses[1].pose.orientation.y, self.path.poses[1].pose.orientation.z, self.path.poses[1].pose.orientation.w]))
            
            change_in_angle = normalize(ENU_to_NED(0,0,math.atan2(target[1] - self.robot_position[1], target[0] - self.robot_position[0]))[2] - ENU_to_NED(0,0,self.robot_position[2])[2])
            self.hsd_msg.heading = min(max(change_in_angle * 180 / math.pi, -90), 90)
            self.hsd_msg.speed = 0.9
            self.hsd_msg.depth = 45.0
            
    # Gets the RRTPlanner path from ROS
    def rrt_path_callback(self, msg):
        self.path = msg
        
    # Gets the most recent robot position from ROS
    def callback_robot_position(self, msg):
        x, y, theta = NED_to_ENU(msg.pose.pose.position.x, msg.pose.pose.position.y, get_yaw_from_quaternion([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]))
        self.robot_position = (x, y, theta)

    # Gets the most recent boolean whether or not to publish a robot command from ROS
    def publish_rrt_hsd_callback(self, msg):
        self.hsd_should_publish = msg.data

# Setup the ROS node
if __name__=='__main__':
    rospy.init_node('rrt_hsd_publisher', log_level=rospy.INFO)
    try:
        hsd_publisher = RRTHSDPublisher()
        rospy.spin()
    except rospy.ROSInterruptException as e:
        print('caught exception:', e)