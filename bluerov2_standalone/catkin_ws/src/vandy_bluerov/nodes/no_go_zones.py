#!/usr/bin/env python

import rospy
import numpy as np
import math
import tf.transformations as trans
import collections
import local_map
from scipy.ndimage import measurements as m
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import Range
from geometry_msgs.msg import Pose, Point
from uuv_gazebo_ros_plugins_msgs.msg import FloatStamped
from std_msgs.msg import Float32MultiArray, Bool
#from scipy iport ndimage as nd
from skimage.draw import (line, polygon)


class NoGoZones(object):
    def __init__(self):
        self.namespace = rospy.get_namespace().replace('/', '')
        rospy.loginfo('Initialize No Go Zones for <%s>' % self.namespace)

       
    def set_no_go_zone(self, obstacle_map, poly, zone_val):
        rr, cc = polygon(poly[:, 0], poly[:, 1], obstacle_map.shape)
        obstacle_map[rr, cc] = zone_val
        rospy.loginfo('No Go Zone maked on map: \n%s' % poly)


if __name__=='__main__':
    rospy.init_node('no_go_zones', log_level=rospy.INFO)
    try:
        node = NoGoZones()
        rospy.spin()
    except rospy.ROSInterruptException:
        print('caught exception')
