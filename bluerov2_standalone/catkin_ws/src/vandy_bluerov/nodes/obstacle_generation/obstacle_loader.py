#!/usr/bin/env python

import rospy
import sys
import numpy as np
import os
import rospkg
from enum import IntEnum

class ObstacleLoader(object):
    def __init__(self, filename):
        rospy.loginfo('Load obstacle list from file for %s', rospy.get_namespace())
        self.node_name_logging = '[Obstacle Loader] '
        self.obstacles = []

        rp = rospkg.RosPack()
        filename = rp.get_path("vandy_bluerov") + "/config/" + filename
        rospy.loginfo(self.node_name_logging + 'filename, file: ' + str(filename))

        if not os.path.isfile(filename):
            rospy.logerr(self.node_name_logging + 'Invalid filename, file: ' + str(filename))
        try:
            with open(filename, 'r') as obstacle_file:
                obstacle_list = yaml.load(obstacle_file)
                if isinstance(obstacle_list, list):
                    for obstacle_data in obstacle_list:
                        obstacle = np.array([
                            obstacle_data['point'][1],
                            obstacle_data['point'][0],
                            obstacle_data['point'][2],
                            obstacle_data['box_velocity_x'],
                            obstacle_data['box_velocity_y'],
                            obstacle_data['box_velocity_z'],
                            obstacle_data['box_size_x'],
                            obstacle_data['box_size_y'],
                            obstacle_data['box_size_z']
                            ], dtype = float)
                        self.add_obstacle(obstacle)
                else:
                    assert 'obstacles' in obstacle_list
                    for obstacle_data in obstacle_list['obstacles']:
                        obstacle = np.array([
                            obstacle_data['point'][1],
                            obstacle_data['point'][0],
                            obstacle_data['point'][2],
                            obstacle_data['box_velocity_x'],
                            obstacle_data['box_velocity_y'],
                            obstacle_data['box_velocity_z'],
                            obstacle_data['box_size_x'],
                            obstacle_data['box_size_y'],
                            obstacle_data['box_size_z']
                            ], dtype = float)
                        self.add_obstacle(obstacle)                    
        except Exception(e):
            rospy.logerr(self.node_name_logging + 'Error while loading the file')
            rospy.logerr(e)
        rospy.loginfo(self.node_name_logging + 'obstacles loaded: %d' %len(self.obstacles))
   
    def get_obstacles(self):
        return self.obstacles
                
    def add_obstacle(self, obstacle):
        # print(obstacle)
        if len(self.obstacles) == 0:
            self.obstacles = [obstacle]
        else:
            self.obstacles = np.append(self.obstacles, [obstacle], axis = 0)
        return True

class ObstacleParams(IntEnum):
    BOX_POS_X = 0
    BOX_POS_Y = 1
    BOX_POS_Z = 2
    BOX_VELOCITY_X = 3
    BOX_VELOCITY_Y = 4
    BOX_VELOCITY_Z = 5
    BOX_SIZE_X = 6
    BOX_SIZE_Y = 7
    BOX_SIZE_Z = 8

if __name__=='__main__':
    print('Starting ObstacleLoader')
    rospy.init_node('obstacle_loader', log_level=rospy.INFO)
    try:
        node = ObstacleLoader()
        rospy.spin()
    except rospy.ROSInterruptException:
        print('caught exception')
