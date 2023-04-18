#!/usr/bin/env python

import rospy
import navpy
import numpy as np
import time
import math
import tf2_ros
import tf, tf.msg
import random
import heading
import geometry_msgs.msg
import tf.transformations as trans

from std_msgs.msg import Float64
from scipy import ndimage as nd
from tf import TransformListener
from nav_msgs.msg import OccupancyGrid, Odometry
from vandy_bluerov.msg import LatLonDepth
# from geometry_msgs.msg import Pose, TransformStamped, Point, Vector3Stamped
from vandy_bluerov.msg import HSDCommand

class LocalMap(object):
    def __init__(self, global_map_size, local_map_size, map_name, map_north_fixed=False):
        self.map_name = map_name
        print("Starting Local Mapping(" + map_name + ") node")

        # Set map rotation behaviour World or Body
        self.map_north_fixed = map_north_fixed

        # Setup the default grid parameters
        self.resolution = 1
        self.global_map_size = global_map_size

        # Setup the subset grid 
        self.grid_msg = OccupancyGrid()
        self.grid_msg.header.frame_id = map_name
        self.grid_msg.info.resolution = self.resolution
        self.grid_msg.info.width = local_map_size
        self.grid_msg.info.height = local_map_size
        self.grid_msg.data = np.arange(local_map_size * local_map_size)


        # Setup the pubs and subs   
        self.odometry_sub = rospy.Subscriber(
             'pose_gt_noisy_ned', Odometry, self.callback_odometry, queue_size=1) 
        self.uuv_position = [0,0,0]
        self.uuv_rpy = [0,0,0]
        self.uuv_yaw = 0

        self.pub = rospy.Publisher(
            "/uuv0/" + map_name, OccupancyGrid, queue_size = 1)      
        self.obstacle_map_local = np.array([], dtype=np.int)

       
        # Setup the coordinate transformation structure
        # self.transform = TransformStamped()

        # listener = TransformListener()
        self.init_grid()

        # rate = rospy.Rate(1)
        
        # tfBuffer = tf2_ros.Buffer(rospy.Duration(5))
        # tfListener = tf2_ros.TransformListener(tfBuffer)

    def publish_local_map(self, local_map):
        # while not rospy.is_shutdown():
        startTime = time.time()
        # TODO: transform this to UUV RPY instead of TF
           
        broadcaster = tf2_ros.StaticTransformBroadcaster()
        static_transformStamped = geometry_msgs.msg.TransformStamped()

        static_transformStamped.header.stamp = rospy.Time.now()
        static_transformStamped.header.frame_id = "world"
        static_transformStamped.child_frame_id = "/" + self.map_name

        static_transformStamped.transform.translation.x = self.uuv_position[1]
        static_transformStamped.transform.translation.y = self.uuv_position[0]
        static_transformStamped.transform.translation.z = -self.uuv_position[2]

        if not self.map_north_fixed:
            quat = tf.transformations.quaternion_from_euler(0,0,1.5708 - math.radians(self.uuv_rpy[2]))
        else:
            quat = tf.transformations.quaternion_from_euler(0,0,1.5708)
        static_transformStamped.transform.rotation.x = quat[0]
        static_transformStamped.transform.rotation.y = quat[1]
        static_transformStamped.transform.rotation.z = quat[2]
        static_transformStamped.transform.rotation.w = quat[3]

        broadcaster.sendTransform(static_transformStamped)
        
        # Set the UUV in the middle of the subset grid
        offset_from_middle = self.grid_msg.info.width // 2 
        self.grid_msg.info.origin.position.x = -offset_from_middle
        self.grid_msg.info.origin.position.y = -offset_from_middle
            
        self.grid_msg.data = local_map.flatten()
        # Publish the grid message
        self.pub.publish(self.grid_msg)
        # print("Grid message published, Time taken: " + str(time.time()-startTime) + " secs")
        # rate.sleep()

    def get_local_map(self, global_map):
        global_map = np.rot90(global_map)
        x = int(-self.uuv_position[1]) - self.grid_msg.info.height//2 + self.global_map_size//2
        y = int(self.uuv_position[0]) - self.grid_msg.info.width//2 + self.global_map_size//2
        # print("x: " + str(x) + " " + str(x+self.grid_msg.info.width) + ", y: " + str(y) + " " + str(y+self.grid_msg.info.width))      
        local_map = np.array(global_map[x:x+self.grid_msg.info.width, y:y+self.grid_msg.info.height])
        if not self.map_north_fixed:
            return nd.rotate(local_map, -self.uuv_rpy[2], reshape=False, order=0)        
        else:
            return local_map

    def vector_to_np(self, v):
        return np.array([v.x, v.y, v.z])
    
    def quaternion_to_np(self, q):
        return np.array([q.x, q.y, q.z, q.w])

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
        rpy = trans.euler_from_quaternion(q, axes='sxyz')
        
        self.uuv_rpy[0] = math.degrees(rpy[0])
        self.uuv_rpy[1] = math.degrees(rpy[1])
        self.uuv_rpy[2] = math.degrees(rpy[2])

    def init_grid(self):
        for i in np.arange(0, self.grid_msg.info.width* self.grid_msg.info.height):
            self.grid_msg.data[i] = int(0)
             
if __name__ == '__main__':
    rospy.init_node('local_map', log_level=rospy.INFO)
    try:
        node = LocalMap()
        rospy.spin()
    except rospy.ROSInterruptException:
        print('caught exception') 

