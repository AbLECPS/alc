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
        rospy.loginfo('Initialize Task: RTH for <%s>' % self.namespace)
        
        # Subscribe to odometry
        self.odometry_sub = rospy.Subscriber(
            'odom', Odometry, self.callback_pose, queue_size=1)    

        # Subscribe to HOME position msg
        self.home_position_sub = rospy.Subscriber(
            '/uuv0/home_position', LatLonDepth, self.callback_home_position, queue_size=1)    
        self.home_position_msg = LatLonDepth() 

        self.hsd_pub = rospy.Publisher(
            '/uuv0/hsd_to_rth', HSDCommand, queue_size=1)   
        self.hsd_cmd = HSDCommand()
        self.hsd_cmd.heading = 0
        self.hsd_cmd.speed = rospy.get_param('~rth_speed', 0.4) 
        self.hsd_cmd.depth = rospy.get_param('~rth_depth', 45) 
        self.max_turnrate = rospy.get_param('~rth_turnrate', 30) 

        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            # ToDo: Check for obstacles Front
            # Publish HSD
            self.hsd_cmd.header.stamp = rospy.Time.now() 
            self.hsd_pub.publish(self.hsd_cmd)            
            rate.sleep()

    def callback_surface_task_enable(self, msg):
        self.surface_task_enable = msg.data        
        
    def callback_pose(self, odom):
        # Convert Quaternion to rpy
        rpy = trans.euler_from_quaternion([odom.pose.pose.orientation.x,
                                     odom.pose.pose.orientation.y,
                                     odom.pose.pose.orientation.z,
                                     odom.pose.pose.orientation.w])        
        self.uuv_position = odom.pose.pose.position
        # rospy.loginfo('UUV: %0.2f, %0.2f, %0.2f' %(self.uuv_position.x,self.uuv_position.y, math.degrees(self.uuv_yaw)))
        self.uuv_yaw = np.degrees(rpy[2])

        # Heading cmd:
        # UUV to Home position gives the heading
        # UUV yaw and heading to Home difference gives Heading CMD
        # Speed and Depth CMD comes from Home message (as RTH alt and speed)
        d_x = self.home_position_msg.latitude - self.uuv_position.x
        d_y = self.home_position_msg.longitude - self.uuv_position.y
        try:
            rth_heading = (math.degrees(math.atan2(d_y,d_x)) + 180) % 360
        except Exception:
            pass

        rth_heading = (rth_heading - self.uuv_yaw + 180) % 360
        if rth_heading > 180:
            rth_heading = -(360 - rth_heading)    
        self.hsd_cmd.heading = max(-self.max_turnrate, min(self.max_turnrate, rth_heading))
        # rospy.loginfo('RTH: %d | UUV %d | HSD %d' %(rth_heading, self.uuv_yaw, self.hsd_cmd.heading))
        # rospy.loginfo('RTH HSD %0.2f' %(self.hsd_cmd.heading))
        


    def callback_home_position(self, msg):
        self.home_position_msg = msg
       
if __name__=='__main__':
    print('Starting  Task: RTH')
    rospy.init_node('task_rth', log_level=rospy.INFO)
    try:
        node = TaskRth()
        rospy.spin()
    except rospy.ROSInterruptException:
        print('caught exception')
