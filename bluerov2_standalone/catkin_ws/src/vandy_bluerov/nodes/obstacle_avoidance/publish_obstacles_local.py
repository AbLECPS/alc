#!/usr/bin/env python
import rospy
from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import Odometry
from nav_msgs.srv import GetMap
import matplotlib.pyplot as plt
from rtreach.msg import reach_tube
from rtreach.msg import interval
import rospkg 
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray


import numpy as np

class PublishObstacles:
    """Generates walls of maps:
       Note: -1: Unknown
              0: Free
              100: Occupied """
    def __init__(self,threshold=0.65,generate_freespace=False,debug=True):
        self.threshold = threshold
        self.intervals = []
        self.markers = []
        self.debug = debug
        self.generate_freespace = generate_freespace
        self.pose_msg = None
        self.odom_sub = rospy.Subscriber("uuv0/pose_gt", Odometry, self.pose_callback, queue_size=1)
        self.pub = rospy.Publisher('obstacles',reach_tube,queue_size=20)
        if(self.debug):
            self.vis_pub =rospy.Publisher('sanity_pub', MarkerArray, queue_size=20)
    
    def pose_callback(self,pose_msg):
        self.pose_msg = pose_msg
        
    def execute(self,msg):
        # map metadata
        if(self.pose_msg):
            origin =[msg.info.origin.position.y+self.pose_msg.pose.pose.position.x,msg.info.origin.position.x+self.pose_msg.pose.pose.position.y]
            res = msg.info.resolution
            map_data= np.asarray(msg.data)
            grid_size=(msg.info.height,msg.info.width)
            if(self.debug):
                rospy.logwarn("grid size ({},{})".format(grid_size[0],grid_size[1]))
            map_data = map_data.reshape(grid_size)
            interval_list = []
            indices = np.where(map_data>0)
            if(self.debug):
                rospy.logwarn("num_indices =({},{})".format(len(indices[0]),len(indices[1])))
            xs = indices[0]
            ys=  indices[1]
            markerArray = MarkerArray()
            for k in range(len(xs)):
                j = xs[k]
                i = ys[k]
                x_point = res*i + origin[0]
                y_point = res*j + origin[1]
                intv = interval()
                intv.x_min = float(x_point) - 0.5 
                intv.x_max = float(x_point) + 0.5
                intv.y_min = float(y_point) - 0.5 
                intv.y_max = float(y_point) + 0.5
                interval_list.append(intv)
                
                if(self.debug):
                    marker = Marker()
                    marker.header.frame_id = "world"
                    marker.header.stamp =rospy.Time.now()
                    marker.id = k
                    marker.type = marker.CUBE
                    marker.action = marker.ADD
                    marker.scale.x =  intv.x_max - intv.x_min
                    marker.scale.y = intv.y_max - intv.y_min
                    marker.scale.z = 1.0
                    marker.color.a = 1.0
                    marker.color.r = 0.0
                    marker.color.g = 1.0
                    marker.color.b = 1.0
                    marker.pose.orientation.w = 1.0
                    marker.pose.position.x = (intv.x_min + intv.x_max) / 2.0 
                    marker.pose.position.y = (intv.y_min + intv.y_max) / 2.0 
                    marker.pose.position.z = self.pose_msg.pose.pose.position.z
                    markerArray.markers.append(marker)

            self.markers = markerArray
            self.intervals = interval_list
        
            
    def publish_reach_tube(self,timer):
        msg = reach_tube()
        msg.obstacle_list = self.intervals
        msg.header.stamp = rospy.Time.now()
        msg.count = len(self.intervals)
        self.pub.publish(msg)
        if(self.debug):
            self.vis_pub.publish(self.markers)

if __name__=="__main__":
    rospy.init_node("publish_obstacles_local")
    args = rospy.myargv()[1:]
    debug = False
    if(len(args)>=1):
        if(type(args[0])==str):
            if(args[0].lower()=="true"):
                debug = True
        else:
            debug = bool(int(args[0]))
    po = PublishObstacles(debug=debug)
    rospy.Subscriber('/uuv0/obstacle_map_local', OccupancyGrid, po.execute, queue_size=10)
    rospy.Timer(rospy.Duration(0.05), po.publish_reach_tube)
    rospy.spin()