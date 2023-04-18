#!/usr/bin/env python
import rospy
from nav_msgs.msg import OccupancyGrid
from nav_msgs.srv import GetMap
from rtreach.msg import reach_tube
from rtreach.msg import interval
import rospkg 


import numpy as np

class PublishObstacles:
    """Generates walls of maps:
       Note: -1: Unknown
              0: Free
              100: Occupied """
    def __init__(self,threshold=0.65,generate_freespace=False):
        self.threshold = threshold
        self.intervals = []
        self.generate_freespace = generate_freespace
        self.pub = rospy.Publisher('obstacles',reach_tube,queue_size=20)
        
       

    def execute(self,msg):
        # map metadata
        origin =[msg.info.origin.position.y,msg.info.origin.position.x]
        res = msg.info.resolution
        map_data= np.asarray(msg.data)
        grid_size=(msg.info.height,msg.info.width)
        map_data = map_data.reshape(grid_size)
        interval_list = []
        indices = np.where(map_data>0)
        xs = indices[0]
        ys=  indices[1]

        for k in range(len(xs)):
            j = xs[k]
            i = ys[k]
            x_point = res*i + origin[0]
            y_point = res*j + origin[1]
            intv = interval()
            intv.x_min = float(x_point)
            intv.x_max = float(x_point)
            intv.y_min = float(y_point)
            intv.y_max = float(y_point)
            interval_list.append(intv)
        self.intervals = interval_list
        
            
    def publish_reach_tube(self,timer):
        msg = reach_tube()
        msg.obstacle_list = self.intervals
        msg.header.stamp = rospy.Time.now()
        msg.count = len(self.intervals)
        self.pub.publish(msg)

if __name__=="__main__":
    rospy.init_node("publish_obstacles")
    po = PublishObstacles()
    rospy.Subscriber('/uuv0/obstacle_map', OccupancyGrid, po.execute, queue_size=10)
    rospy.Timer(rospy.Duration(0.05), po.publish_reach_tube)
    rospy.spin()