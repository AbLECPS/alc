#!/usr/bin/env python



import rospy
import rospy
import numpy as np
import math
from std_srvs.srv import Empty
from vandy_bluerov.msg import HSDCommand
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from rospy.numpy_msg import numpy_msg

from uuv_gazebo_ros_plugins_msgs.msg import FloatStamped
from std_msgs.msg import Float64
from std_msgs.msg import Bool
from vandy_bluerov.msg import LatLonDepth
from message_filters import ApproximateTimeSynchronizer, Subscriber
import tf.transformations as trans

class TaskRth(object):
    def __init__(self):
        self.namespace = rospy.get_namespace().replace('/', '')
        rospy.loginfo('Initialize Task: Surface for <%s>' % self.namespace)
        
        # Subscribe to odometry
        self.odometry_sub = rospy.Subscriber(
            'odom', Odometry, self.callback_pose, queue_size=1)    
        self.uuv_depth = 0.0

        # # Subscribe to HOME position msg
        # self.home_position_sub = rospy.Subscriber(
        #     'home_position', LatLonDepth, self.surface_task_enable_callback, queue_size=1)    

        self.hsd_pub = rospy.Publisher('/uuv0/hsd_to_surface', HSDCommand, queue_size=1)   
        self.hsd_cmd = HSDCommand()
        self.hsd_cmd.heading = rospy.get_param('~surface_helix_turnrate', 30) 
        self.hsd_cmd.speed = 0.4 

        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            # if self.surface_task_enable:
            # Check for obstacles Front and Upwards
            # Odometry is needed for this (UUV yaw)
            # ...
            # Publish HSD
            self.hsd_cmd.header.stamp = rospy.Time.now() 
            self.hsd_cmd.depth = self.uuv_depth - 50 
            self.hsd_pub.publish(self.hsd_cmd)            
            rate.sleep()   
        
    def callback_pose(self, odom):
        # Convert Quaternion to rpy
        rpy = trans.euler_from_quaternion([odom.pose.pose.orientation.x,
                                     odom.pose.pose.orientation.y,
                                     odom.pose.pose.orientation.z,
                                     odom.pose.pose.orientation.w])        
        self.uuv_depth = odom.pose.pose.position.z
        # self.uuv_yaw = np.degrees(rpy[2])
        # rospy.loginfo('UUV: %0.2f, %0.2f, %0.2f' %(self.uuv_position.x,self.uuv_position.y, math.degrees(self.uuv_yaw)))


       
if __name__=='__main__':
    print('Starting Task: Surface')
    # rospy.init_node('task_surface', log_level=rospy.DEBUG)
    rospy.init_node('task_surface', log_level=rospy.INFO)
    try:
        node = TaskRth()
        rospy.spin()
    except rospy.ROSInterruptException:
        print('caught exception')
