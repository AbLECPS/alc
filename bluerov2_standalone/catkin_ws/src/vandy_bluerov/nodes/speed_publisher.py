#!/usr/bin/env python

import rospy
import numpy as np
import math
import tf.transformations as trans

from std_msgs.msg import Float32
from nav_msgs.msg import Odometry
from vandy_bluerov.srv import GoToHSD



class UUVSpeedPublisher(object):
    def __init__(self):
        self.namespace = rospy.get_namespace().replace('/', '')
        rospy.loginfo('Initialize UUVSpeedPublisher for <%s>' % self.namespace)

        self.odometry_sub = rospy.Subscriber(
            'odom', Odometry, self.odometry_callback, queue_size=1)    
    
        self.pub = rospy.Publisher(
            'speed', Float32, queue_size=1)

    @staticmethod
    def unwrap_angle(t):
        return math.atan2(math.sin(t),math.cos(t))

    @staticmethod
    def vector_to_np(v):
        return np.array([v.x, v.y, v.z])

    @staticmethod
    def quaternion_to_np(q):
        return np.array([q.x, q.y, q.z, q.w])

    @staticmethod
    def vector_to_mag(v):
        return np.linalg.norm(np.array([v.x, v.y, v.z]))

    @staticmethod
    def degree_to_rad(ang):
        return ang*np.pi/180.


    def odometry_callback(self, msg):
        vel = self.vector_to_mag(msg.twist.twist.linear)                                
        self.pub.publish(vel)                        
        



if __name__=='__main__':
    print('Starting UUVSpeedPublisher')
    rospy.init_node('UUVSpeedPublisher')
    try:
        node = UUVSpeedPublisher()
        rospy.spin()
    except rospy.ROSInterruptException:
        print('caught exception')
