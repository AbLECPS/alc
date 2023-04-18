#!/usr/bin/env python
import rospy
from nav_msgs.msg import OccupancyGrid
from nav_msgs.srv import GetMap
import matplotlib.pyplot as plt
from rtreach.msg import reach_tube
from rtreach.msg import interval
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
import rospkg 

from geometry_msgs.msg import Pose
import tf2_ros
# import tf2_geometry_msgs

"""Credit for Transforms: https://answers.ros.org/question/323075/transform-the-coordinate-frame-of-a-pose-from-one-fixed-frame-to-another/"""

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
        self.pub = rospy.Publisher('obstacles',reach_tube,queue_size=1)
        self.debug = False
        self.rviz = True
        self.local = True
        if(self.rviz):
            self.publisher = rospy.Publisher("local_obstacle_list", MarkerArray, queue_size="1")

        if(self.local):
            rospy.Subscriber('/uuv0/obstacle_map_local', OccupancyGrid, self.execute, queue_size=1)

            # needed to convert from local to global frame
            self.tf_buffer = tf2_ros.Buffer()
            self.listener = tf2_ros.TransformListener(self.tf_buffer)
        else:
            rospy.Subscriber('/uuv0/obstacle_map', OccupancyGrid, self.execute, queue_size=1)

        rospy.Timer(rospy.Duration(0.05), self.publish_reach_tube)


    def transform_pose(self,input_pose, from_frame, to_frame):


        pose_stamped = Pose()#tf2_geometry_msgs.PoseStamped()
        pose_stamped.pose = input_pose
        pose_stamped.header.frame_id = from_frame
        pose_stamped.header.stamp = rospy.Time.now()

        try:
            # ** It is important to wait for the listener to start listening. Hence the rospy.Duration(1)
            output_pose_stamped = self.tf_buffer.transform(pose_stamped, to_frame, rospy.Duration(0.1))
            return output_pose_stamped.pose

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            print(e)
        
       

    def execute(self,msg):
        # map metadata
        origin =[msg.info.origin.position.y,msg.info.origin.position.x]
        if(self.debug):
            rospy.logwarn("origin ({},{}), width: ({} {})".format(origin[0],origin[1],msg.info.height,msg.info.width))
        res = msg.info.resolution
        map_data= np.asarray(msg.data)
        grid_size=(msg.info.height,msg.info.width)
        map_data = map_data.reshape(grid_size)
        interval_list = []
        indices = np.where(map_data>0)
        if(len(indices[0])>0 and self.debug):
            rospy.logwarn("Map Updated")
        xs = indices[0]
        ys=  indices[1]


        markerArray = MarkerArray()

        k = 0 
        while k < len(xs):
            j = xs[k]
            i = ys[k]
            x_point = res*i + origin[0]
            y_point = res*j + origin[1]

            if(self.local):
                my_pose = Pose()
                my_pose.position.x = x_point
                my_pose.position.y = y_point
                my_pose.position.z = 0
                my_pose.orientation.x = 0.0
                my_pose.orientation.y = 0.0
                my_pose.orientation.z = 0.0
                my_pose.orientation.w = 1.0

                transformed_pose = self.transform_pose(my_pose, "obstacle_map_local", "world")
                x_point = transformed_pose.position.x
                y_point = transformed_pose.position.y

            if(self.debug):
                 rospy.logwarn("Transformed Pose x : {}, y: {}".format(y_point,x_point))
            
            
            intv = interval()
            intv.y_min = float(x_point) - 1.0
            intv.y_max = float(x_point) + 1.0
            intv.x_min = float(y_point) - 1.0
            intv.x_max = float(y_point) + 1.0
            intv.value = map_data[j][i]
            interval_list.append(intv)

            

           
            
            if(self.rviz):
                x = (intv.x_max + intv.x_min) / 2
                y = (intv.y_max + intv.y_min) / 2
                marker = Marker()
                marker.header.frame_id = "world"
                marker.type = marker.CUBE
                marker.action = marker.ADD
                marker.scale.x = intv.x_max - intv.x_min
                marker.scale.y = intv.y_max - intv.y_min
                marker.scale.z = 1.1
                marker.color.a = 1.0
                marker.color.r = 0.0
                marker.color.g = 0.0
                marker.color.b = 1.0
                marker.pose.orientation.w = 1.0
                marker.pose.position.x = y
                marker.pose.position.y = x
                marker.pose.position.z = -30.0
                marker.id = k 
                markerArray.markers.append(marker)
            
            if(self.debug):
                rospy.logwarn(intv.value)
            
            
            k+=1

        interval_list.sort(key=lambda x: x.value, reverse=True)
        self.intervals = interval_list


        if(self.rviz):
            self.publisher.publish(markerArray)
        
        
        
       
        
            
    def publish_reach_tube(self,timer):
        msg = reach_tube()
        msg.obstacle_list = self.intervals
        msg.header.stamp = rospy.Time.now()
        msg.count = len(self.intervals)
        self.pub.publish(msg)

if __name__=="__main__":
    rospy.init_node("publish_obstacles")
    po = PublishObstacles()
    rospy.spin()
