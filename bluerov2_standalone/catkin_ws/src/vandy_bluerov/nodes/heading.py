#!/usr/bin/env python

import rospy
import numpy as np
import math
import tf.transformations as trans

class HeadingCalculator(object):
    def __init__(self, max_turnrate=30):
        self.max_turnrate = max_turnrate
        self.xtrack_error = 0
            
    def get_yaw(self, odom):
        rpy = trans.euler_from_quaternion([odom.pose.pose.orientation.x,
                                     odom.pose.pose.orientation.y,
                                     odom.pose.pose.orientation.z,
                                     odom.pose.pose.orientation.w])     
        return np.degrees(rpy[2])
    
    def get_position(self, odom):
        return self.vector_to_np(odom.pose.pose.position)

    def get_heading(self, start, goal):        
        d_x = goal[1] - start[0]
        d_y = goal[0] - start[1]
        try:
            heading = math.degrees(math.atan2(d_y,d_x))
        except Exception:
            return 0
        return heading

    def get_wp_heading(self, start, goal):        
        d_x = goal[1] - start[1]
        d_y = goal[0] - start[0]
        try:
            heading = math.degrees(math.atan2(d_y,d_x))
        except Exception:
            return 0
        return heading

    def get_heading_cmd(self, odom, goal):
        heading = self.get_heading_diff(self.get_heading(self.get_position(odom), goal), self.get_yaw(odom))
        return self.limit_turnrate(heading)

    def get_heading_diff(self, heading, yaw):            
        return (heading - yaw + 180) % 360 - 180

    def limit_turnrate(self, heading):    
        return max(-self.max_turnrate, min(self.max_turnrate, heading))
        
    def planar_distance(self, base, pos):
        return np.sqrt((base[0] - pos[0])**2 +
                       (base[1] - pos[1])**2)

    def vector_to_np(self, v):
        return np.array([v.x, v.y, v.z])

    def get_loiter_heading_cmd(self, odom, goal, radius):
        K_p = 10
        start = self.get_position(odom)
        distance_error = self.planar_distance([start[1], start[0]], goal) - radius
        heading = self.get_heading(self.get_position(odom), goal)
        modification = 90 - min(abs(distance_error * K_p ), 90) * np.sign(distance_error)
        loiter_heading = self.get_heading_diff(heading + modification, self.get_yaw(odom))
        return self.limit_turnrate(loiter_heading)
        
    def get_cross_track_distance(self, odom, start, goal):
        R = 6371000
        pos = self.get_position(odom)
        angular_distance = self.planar_distance([pos[1], pos[0]], start) / R
        wp_heading = np.radians(self.get_heading(self.get_position(odom), start))
        path_heading = np.radians(self.get_heading([goal[1], goal[0]], start))
        xte = np.arcsin(np.sin(angular_distance) * np.sin(wp_heading - path_heading)) * R
        return xte

    def get_path_heading_cmd(self, odom, start, goal):
        K_d = 3
        K_c = 25.0
        pos = self.get_position(odom)
        wp_distance = self.planar_distance([pos[1], pos[0]], goal)
        self.xtrack_error = self.get_cross_track_distance(odom, start, goal)
        wp_heading = self.get_heading(self.get_position(odom), goal)
        path_heading = self.get_heading([start[1], start[0]], goal)
        base = abs(self.xtrack_error * K_c * np.radians(self.get_heading_diff(path_heading, wp_heading))) * np.sign(self.xtrack_error) 
        path_mod = self.limit_turnrate(min(50, wp_distance) * 0.02 * np.power(base, K_d))
        heading = self.get_heading_diff(wp_heading, self.get_yaw(odom))
        # print(str(round(heading,1)) + ' ' + str(round(path_mod,1)))
        heading_cmd = self.limit_turnrate(heading - path_mod)
        return heading_cmd




    
