#! /usr/bin/env python
import rospy
import numpy as np
from gazebo_ros import gazebo_interface
from gazebo_msgs.msg import *
import tf.transformations as tft
from geometry_msgs.msg import Point, Pose, Quaternion, Twist, Wrench
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import Point

from rtreach.msg import interval 
from rtreach.msg import reach_tube

"""
PyMap3D provides coordinate transforms and geodesy functions with a similar API to the Matlab Mapping Toolbox, but was of course independently derived.
"""
try:
    import pymap3d
    use_pymap = True
except Exception as ex:
    print('Package pymap3d is not available, WGS84 coordinates cannot be used\n'
          'Download pymap3d for Python 2.7 as\n'
          '>> sudo pip install pymap3d==1.5.2\n'
          'or for Python 3.x as\n'
          '>> sudo pip install pymap3d')


class GeoFencing:
    def __init__(self,initial_lat_lon_alt,width=350):
        self.initial_xyz = [0,0,0]
        self.initial_rpy = [0,0,0]
        self.initial_geo = [initial_lat_lon_alt[0],initial_lat_lon_alt[1],initial_lat_lon_alt[2]]
        self.ref_geo     = [initial_lat_lon_alt[0],initial_lat_lon_alt[1],initial_lat_lon_alt[2]]

        rospy.loginfo('Using the geodetic coordinates to spawn the model')
        rospy.loginfo('Geodetic coordinates: (%.7f, %.7f, %.2f)',self.initial_geo[0], self.initial_geo[1],self.initial_geo[2])
        rospy.loginfo('Geodetic reference: (%.7f, %.7f, %.2f)',self.ref_geo[0], self.ref_geo[1], self.ref_geo[2])
        
        enu_pos = pymap3d.geodetic2enu(
                self.initial_geo[0], self.initial_geo[1], self.initial_geo[2],self.ref_geo[0], self.ref_geo[1], self.ref_geo[2])

        initial_pose = Pose()
        initial_pose.position.x = enu_pos[0]
        initial_pose.position.y = enu_pos[1]
        initial_pose.position.z = enu_pos[2]
        rospy.loginfo('Initial position wrt the world frame:'' (%.2f, %.2f, %.2f)' %(initial_pose.position.x, initial_pose.position.y,initial_pose.position.z))


        # define the points of a bounding box
        c1,c2,c3,c4 = self.generate_corners(initial_pose.position.x,initial_pose.position.x,width)
        
        line1 = [c1[0],c1[1],c2[0],c2[1]]
        line2 = [c2[0],c2[1],c4[0],c4[1]]
        line3 = [c4[0],c4[1],c3[0],c3[1]]
        line4 = [c3[0],c3[1],c1[0],c1[1]]

        # define intervals that will be published to the rtreach nodes 
        # for safety checking
        int1 = self.generate_interval(line1)
        int2 = self.generate_interval(line2)
        int3 = self.generate_interval(line3)
        int4 = self.generate_interval(line4)

        self.line_intervals = [int1,int2,int3,int4]
        
        self.lines = [line1,line2,line3,line4]

        self.vis_pub = rospy.Publisher('uuv0/bounding_box', MarkerArray,queue_size=1)
        self.reach_pub = rospy.Publisher('uuv0/bounding_box_interval', reach_tube,queue_size=1)

    
    # generates an interval based on a line
    def generate_interval(self,line):
        
        x_min = min(line[0],line[2]) -0.1
        x_max = max(line[0],line[2]) +0.1
        y_min = min(line[1],line[3]) -0.1
        y_max = max(line[1],line[3]) +0.1

        new_interval = interval()
        new_interval.x_min = x_min
        new_interval.x_max = x_max
        new_interval.y_min = y_min
        new_interval.y_max = y_max
        return new_interval

    # this generates a square, we can figure out how to rotate n all that later
    def generate_corners(self,center_x,center_y,width):

        # top left 
        corner1 = (center_x - width/2,center_y+width/2)
        # top right
        corner2 = (center_x + width/2,center_y+width/2)
        # bottom left
        corner3 = (center_x - width/2,center_y-width/2)
        # bottom right
        corner4 = (center_x + width/2,center_y-width/2)
        return corner1, corner2, corner3, corner4


    def visualize_lines(self):
        
        markerArray = MarkerArray()
        for i in range(len(self.lines)):
            line = self.lines[i]
            x1,y1,x2,y2 = line

            marker = Marker()
            marker.id = 10000 + i
            marker.header.frame_id = "world"
            marker.type = marker.LINE_STRIP
            marker.action = marker.ADD
            
            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            
            marker.scale.x = 0.3
            marker.scale.y = 0.3
            marker.scale.z = 0.3


            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0

            marker.points = []
            first_line_point = Point()
            first_line_point.x = x1
            first_line_point.y = y1
            first_line_point.z = -45
            marker.points.append(first_line_point)

            # second point
            second_line_point = Point()
            second_line_point.x = x2
            second_line_point.y = y2
            second_line_point.z = -45
            marker.points.append(second_line_point)
            
            markerArray.markers.append(marker)

        reach_set = reach_tube()
        reach_set.obstacle_list = self.line_intervals
        reach_set.header.stamp = rospy.Time.now()
        reach_set.count = len(self.line_intervals)

        self.vis_pub.publish(markerArray)
        self.reach_pub.publish(reach_set)

if __name__ == '__main__':

    rospy.init_node('geofencing_node')
    args = rospy.myargv()[1:]

    # get the latitude longitude and alitude from the launch file
    latitude=float(args[0])
    longitude=float(args[1])
    altitude=float(args[2])


    gn = GeoFencing([latitude,longitude,altitude])
    r = rospy.Rate(80)
    while not rospy.is_shutdown():
        r.sleep()
        gn.visualize_lines()


    