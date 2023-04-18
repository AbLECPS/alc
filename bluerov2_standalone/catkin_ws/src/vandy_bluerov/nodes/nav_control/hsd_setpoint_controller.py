#!/usr/bin/env python
import os
import sys
import rospy
import numpy as np
import math
import tf.transformations as trans

from vandy_bluerov.msg import HSDCommand
from nav_msgs.msg import Odometry
from vandy_bluerov.srv import GoToHSD

class HSDtoBlueROV2WaypointSRV(object):
    def __init__(self):
        self.namespace = rospy.get_namespace().replace('/', '')
        rospy.loginfo('Initialize control for vehicle <%s>' % self.namespace)

        self.base_link = rospy.get_param('~base_link', 'base_link')

        # Reading the minimum and maximum speed
        self.min_speed = rospy.get_param('~min_speed', 0)
        assert self.min_speed >= 0
        self.max_speed = rospy.get_param('~max_speed', 1.5)
        assert self.max_speed > self.min_speed

        self.uuv_heading=0

        # Get set of initial heading, speed, and depth
        init_heading = rospy.get_param('~init_heading', 0.0)
        init_speed = rospy.get_param('~init_speed', 2.0)
        init_depth = rospy.get_param('~init_depth', 22)

        self.uuv_degradation_mode = rospy.get_param('~uuv_degradation_mode', 'x')

        self.hsd_command = dict(heading = init_heading,
                                speed = init_speed,
                                depth = init_depth)

        # Wait for other nodes to init before starting any movement
        while rospy.Time.now() < rospy.Time(1):
            pass
        rospy.loginfo("### Full steam ahead ###")

        # Subscribe to AUV Heading, Speed, and Depth command
        self.hsd_command_sub = rospy.Subscriber(
            'hsd_command', HSDCommand, self.update_HSD, queue_size=1)

        self.odometry_sub = rospy.Subscriber(
            'odom', Odometry, self.odometry_callback, queue_size=1)    
    
        self.error_pub = rospy.Publisher(
            'error', HSDCommand, queue_size=1)

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

    def update_HSD(self, msg):
        '''
            Receives HSD command and call interpreter service on BlueROV2
        '''
        # Limit speed
        self.hsd_command = dict(heading = self.uuv_heading + self.degree_to_rad(msg.heading), #heading + heading_change
                                speed   = min(max(self.min_speed,msg.speed), self.max_speed),
                                depth   = msg.depth)

        try:
            rospy.wait_for_service('go_to_HSD', timeout=5)
        except rospy.ROSException:
            raise rospy.ROSException('Service not available! Closing node...')

        try:
            go_to_HSD_srv = rospy.ServiceProxy(
                'go_to_HSD',
                GoToHSD)
        except rospy.ServiceException(e):
            raise rospy.ROSException('Service call failed, error=' + e)
        
        # UUV modes of operation of degradation
        if self.uuv_degradation_mode == 4:
            self.hsd_command['heading'] += 0.01
            # rospy.loginfo('Mode 4 UUV degradation mode')
        elif self.uuv_degradation_mode == 5:
            self.hsd_command['heading'] += 0.05
            # rospy.loginfo('Mode 5 UUV degradation mode')
                
        success = go_to_HSD_srv(
                            self.hsd_command['heading'],
                            self.hsd_command['speed'],
                            self.hsd_command['depth'])
        # rospy.loginfo('HSD: %0.2f | %0.2f | %d' %(self.hsd_command['heading'], self.hsd_command['speed'],self.hsd_command['depth']))

        if not success:
            rospy.loginfo('Failed to call HSD service')

    def odometry_callback(self, msg):
        """Calculate & publish errors"""

        pos = [msg.pose.pose.position.x,
               msg.pose.pose.position.y,
               msg.pose.pose.position.z]

        quat = [msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z,
                msg.pose.pose.orientation.w]

        # Calculate the position, position, and time of message
        p = self.vector_to_np(msg.pose.pose.position)
        vel = self.vector_to_mag(msg.twist.twist.linear)

        q = self.quaternion_to_np(msg.pose.pose.orientation)
        rpy = trans.euler_from_quaternion(q, axes='sxyz')
        self.uuv_heading = rpy[2]

        # Calculate error for heading
        errors = self.calculate_errors(vel, p, rpy)

        hsd_error_msg = HSDCommand(header = msg.header,
                         heading = errors['heading'],
                         speed = errors['speed'],
                         depth = errors['depth'])
                                        
        self.error_pub.publish(hsd_error_msg)                        
        
    def calculate_errors(self, vel, pos, rpy):
        """
            Calculates the errors to feed into the four PID controllers
            Input:
                vel = velocity
                pos = position (x, y, z)
                rpy = orientation (roll, pitch, yaw)
            Output:
                Dictionary containing errors for heading, roll, pitch, and speed
        """
        errors = {}
        # Calculate error in yaw
        errors['heading'] = self.unwrap_angle(self.hsd_command['heading'] - rpy[2])
        
        # Calcualte error for speed
        errors['speed'] = self.hsd_command['speed'] - vel

        # Calculate error in depth
        errors['depth'] = self.hsd_command['depth'] - pos[2]
        
        return errors


if __name__=='__main__':
    print('Starting HSD to BlueROV2 waypoint interpreter service')
    rospy.init_node('hsd_to_bluerov2_waypoint_srv')
    try:
        node = HSDtoBlueROV2WaypointSRV()
        rospy.spin()
    except rospy.ROSInterruptException:
        print('caught exception')
