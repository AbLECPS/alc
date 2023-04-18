#!/usr/bin/env python
import rospy
import numpy as np
import math
import tf.transformations as trans

from std_msgs.msg import Float32, Float32MultiArray
from nav_msgs.msg import Odometry


class ObstacleDistancePublisher(object):
    def __init__(self):

        # Odom/Pose message
        self.odometry_sub = rospy.Subscriber(
             'uuv0/pose_gt_noisy_ned', Odometry, self.callback_odometry, queue_size=1) 
        self.uuv_position = None

        self.obstacle_sub = rospy.Subscriber(
             'box0/pose_gt', Odometry, self.callback_obstacle, queue_size=1) 
        self.box_positions = []
    
        self.pub = rospy.Publisher(
            'obstacle_distances_gt', Float32MultiArray, queue_size=1)
        self.obstacle_distances = []

    @staticmethod
    def vector_to_np(v):
        return np.array([v.x, v.y, v.z])

    def callback_obstacle(self, msg):
        # Get obstacle position
        # Coordinate system change
        # ENU -> NED
        pos = [msg.pose.pose.position.y,
               msg.pose.pose.position.x,
               msg.pose.pose.position.z]
        if pos not in self.box_positions:
            self.box_positions.append(pos)
            # print('*****************************')
            # print(self.box_positions)

    def callback_odometry(self, msg):
        # Get UUV position
        pos = [msg.pose.pose.position.x,
               msg.pose.pose.position.y,
               -msg.pose.pose.position.z]
        # Calculate the position, position, and time of message
        self.uuv_position = self.vector_to_np(msg.pose.pose.position)
        # If there is at least one valid obstacle:
        if len(self.box_positions) > 1:
            self.obstacle_distances = []
            # Drop first element - faulty from init time
            for box_pos in self.box_positions[1:]:
                distance_3d = np.sqrt((box_pos[0] - pos[0])**2 + \
                                      (box_pos[1] - pos[1])**2 + \
                                      (box_pos[2] - pos[2])**2)
                self.obstacle_distances.append(distance_3d)
        else:
            self.obstacle_distances = []
        # Publish GT distances
        self.pub.publish(Float32MultiArray(data = np.array(self.obstacle_distances).flatten())) 

if __name__=='__main__':
    print('Starting ObstacleDistancePublisher')
    rospy.init_node('ObstacleDistancePublisher')
    try:
        node = ObstacleDistancePublisher()
        rospy.spin()
    except rospy.ROSInterruptException:
        print('caught exception')
