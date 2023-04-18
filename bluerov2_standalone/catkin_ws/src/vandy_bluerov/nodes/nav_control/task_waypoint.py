#!/usr/bin/env python

import rospy
import rospy
import numpy as np
import math
import tf.transformations as trans
import waypoint_actions
import heading

from std_srvs.srv import Empty
from vandy_bluerov.msg import HSDCommand
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from rospy.numpy_msg import numpy_msg
from uuv_gazebo_ros_plugins_msgs.msg import FloatStamped
from std_msgs.msg import Float64, Int32
from std_msgs.msg import Bool
from vandy_bluerov.msg import LatLonDepth
from message_filters import ApproximateTimeSynchronizer, Subscriber
from std_msgs.msg import String, Time, Float64MultiArray
from geometry_msgs.msg import Point

class TaskWaypoint(object):
    def __init__(self):
        self.namespace = rospy.get_namespace().replace('/', '')
        rospy.loginfo('Initialize Task: Waypoint for <%s>' % self.namespace)
        
        # CM hsd input msg
        self.cm_hsd_input_sub= rospy.Subscriber(
            "/uuv0/cm_hsd_input", String, self.callback_cm_hsd_input)
        self.cm_hsd_input =  String()

        # Subscribe to odometry
        self.odometry_sub = rospy.Subscriber(
            '/uuv0/pose_gt_noisy_ned', Odometry, self.callback_pose, queue_size=1)    

        self.waypoint_pub = rospy.Subscriber(
            '/uuv0/waypoints', Float64MultiArray, self.callback_waypoint, queue_size=1)   
        self.waypoints = []
        self.target_waypoint_id = 0

        # self.obstacle_map_sub = rospy.Subscriber(
        #     "/uuv0/obstacle_map", OccupancyGrid, self.callback_obstacle_map)
        
        # self.obstacle_map = np.array([], dtype=np.int)
        # self.map_size = 1000 #default, update it from msg
        # self.map_origin = Pose()

        # Subscribe to collision avoidance
        self.hsd_obstacle_avoidance_sub = rospy.Subscriber(
            '/uuv0/hsd_obstacle_avoidance', HSDCommand, self.obstacle_avoidance_callback)
        self.hsd_obstacle_avoidance_msg = HSDCommand()

        # self.obstacle_near_wp_sub = rospy.Subscriber(
        #     "/uuv0/obstacle_near_wp", Bool, self.callback_next_wp, queue_size = 1)

        self.next_wp_sub = rospy.Subscriber(
            "/uuv0/next_wp", Bool, self.callback_next_wp, queue_size = 1)

        self.waypoints_completed_pub = rospy.Publisher(
            '/uuv0/waypoints_completed', Bool, queue_size=1)

        self.new_wp_pub = rospy.Publisher(
            '/uuv0/new_waypoint', Point, queue_size=1)    

        self.target_pub = rospy.Publisher(
            '/uuv0/target_waypoint', Point, queue_size=1)   
        self.target_id_pub = rospy.Publisher(
            '/uuv0/target_waypoint_id', Int32, queue_size=1)   
        self.hsd_pub = rospy.Publisher(
            '/uuv0/hsd_to_waypoint', HSDCommand, queue_size=1)   
        self.waypoint_distance_pub = rospy.Publisher(
            '/uuv0/distance_to_waypoint', Float64, queue_size=1)   
        self.obstacle_avoidance_direction_pub = rospy.Publisher(
            '/uuv0/obstacle_avoidance_direction', Float64, queue_size=1)    
                       
        self.xtrack_error_pub = rospy.Publisher(
            '/uuv0/xtrack_error', Float64, queue_size=1)   
        self.xtrack_error_pub.publish(Float64(float('nan')))
        
        self.wa = waypoint_actions.WaypointAction
        self.wp = waypoint_actions.WaypointParams
        self.hc = heading.HeadingCalculator()

        self.hsd_cmd = HSDCommand()
        self.hsd_cmd.heading = 0
        self.max_turnrate = rospy.get_param('~max_turnrate', 30) 
        self.waypoint_radius = rospy.get_param('~waypoint_radius', 5)
        self.random_waypoints = rospy.get_param('~random_waypoints', False)
        self.last_waypoint_created = rospy.Time.now()
        np.random.seed(rospy.get_param('~random_seed', 0))

        self.num_waypoints = rospy.get_param('~num_waypoints', 5) 
        self.min_distance = rospy.get_param('~min_distance', 15) # meter 
        self.max_distance = rospy.get_param('~max_distance', 30) # meter
        self.min_heading = rospy.get_param('~min_heading', -1.5708) # rad
        self.max_heading = rospy.get_param('~max_heading', 1.5708) # rad
        self.xte = rospy.get_param('~x_track_error', False) # rad

        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            if self.cm_hsd_input.data == 'waypoint_task':
                # Publish HSD
                self.hsd_cmd.header.stamp = rospy.Time.now() 
                h = self.hsd_cmd
                self.hsd_pub.publish(h)            
                # print('\t\t\t\t\t\tHSD sent:     ' + str(rospy.get_time()) + ' ' +  str(h.heading) + ' ' + str(h.header.seq))
            rate.sleep()
    
    def callback_cm_hsd_input(self, msg):
        self.cm_hsd_input = msg

    def callback_waypoint(self, msg):
        self.waypoints = np.reshape(msg.data,(-1, 7))
        # if len(self.waypoints) > 0:
        #     self.target_waypoint_id = 0   
        # rospy.loginfo(self.waypoints)
        
    def callback_pose(self, odom):
        if self.cm_hsd_input.data == 'waypoint_task':
            # Convert Quaternion to rpy
            rpy = trans.euler_from_quaternion([odom.pose.pose.orientation.x,
                                        odom.pose.pose.orientation.y,
                                        odom.pose.pose.orientation.z,
                                        odom.pose.pose.orientation.w])        
            self.uuv_position = self.vector_to_np(odom.pose.pose.position)
            self.uuv_yaw = np.degrees(rpy[2])
            # Startup delay
            if rospy.Time.now() > rospy.Time(10):
                if self.target_waypoint_id < len(self.waypoints):
                    if self.target_waypoint_id != -1:
                        target = Point()
                        target.x = self.waypoints[self.target_waypoint_id][self.wp.X]
                        target.y = self.waypoints[self.target_waypoint_id][self.wp.Y]
                        target.z = self.waypoints[self.target_waypoint_id][self.wp.Z]
                        self.target_pub.publish(target)
                        # print(str(target.x) + " " + str(target.y) + " " + str(target.z))
                        self.target_id_pub.publish(Int32(self.target_waypoint_id))
                        
                        # Calculate and publish next WP direction for obstacle avoidance 
                        if self.target_waypoint_id + 1 < len(self.waypoints):
                            # next_heading = self.hc.get_heading_cmd(
                            #     odom, 
                            #     self.waypoints[self.target_waypoint_id+1][self.wp.X:2])                            
                            next_heading = self.hc.get_heading_diff(
                                self.hc.get_heading(self.hc.get_position(odom), self.waypoints[self.target_waypoint_id][self.wp.X:2]), 
                                self.hc.get_heading(self.hc.get_position(odom), self.waypoints[self.target_waypoint_id+1][self.wp.X:2])
                            )
                            self.obstacle_avoidance_direction_pub.publish(Float64(next_heading))
                        
                        # Calculate and publish UUV distance to WP
                        distance = self.dist(self.waypoints[self.target_waypoint_id][:])
                        self.waypoint_distance_pub.publish(Float64(distance))
                        
                        # Calculate and publish Cross track error
                        if (self.target_waypoint_id > 0):
                            xtrack_error = self.hc.get_cross_track_distance(
                                        odom, 
                                        self.waypoints[self.target_waypoint_id - 1][self.wp.X:2],
                                        self.waypoints[self.target_waypoint_id][self.wp.X:2]
                                    )
                            self.xtrack_error_pub.publish(Float64(xtrack_error))

                        # Set HSD values
                        # If normal 'tuch and go' type:
                        if self.waypoints[self.target_waypoint_id][self.wp.ACTION] == self.wa.PASS:
                            if (self.xte):
                                # Cross track error minimalisation:
                                if (self.target_waypoint_id > 0):
                                    self.hsd_cmd.heading = self.hc.get_path_heading_cmd(
                                        odom, 
                                        self.waypoints[self.target_waypoint_id - 1][self.wp.X:2],
                                        self.waypoints[self.target_waypoint_id][self.wp.X:2]
                                    )
                                else:
                                    self.hsd_cmd.heading = 0
                            else:                            
                                # No cross track error minimalisation:    
                                self.hsd_cmd.heading = self.hc.get_heading_cmd(
                                    odom, 
                                    self.waypoints[self.target_waypoint_id][self.wp.X:2])
                            self.hsd_cmd.speed = self.waypoints[self.target_waypoint_id][self.wp.SPEED]
                            self.hsd_cmd.depth = self.waypoints[self.target_waypoint_id][self.wp.Z]
                            # Update target WP no.:
                            if (distance < self.waypoint_radius):
                                self.target_waypoint_id = self.target_waypoint_id + 1    
                        # If loiter type for FDR Search and rescue:
                        elif self.waypoints[self.target_waypoint_id][self.wp.ACTION] == self.wa.LOINTER_N:                    
                            self.hsd_cmd.heading = self.hc.get_loiter_heading_cmd(
                                odom, 
                                self.waypoints[self.target_waypoint_id][self.wp.X:2],
                                self.waypoints[self.target_waypoint_id][self.wp.P1])
                            self.hsd_cmd.speed = self.waypoints[self.target_waypoint_id][self.wp.SPEED]
                            self.hsd_cmd.depth = self.waypoints[self.target_waypoint_id][self.wp.Z]
                            # Todo:
                            # if PROCESS TURNS ...
                            # self.waypoints[self.target_waypoint_id][self.wp.P0] 
                                # self.target_waypoint_id = self.target_waypoint_id + 1    :
                            # Publish loiter direction for obstacle avoidance
                            self.obstacle_avoidance_direction_pub.publish(self.waypoints[self.target_waypoint_id][self.wp.P1])
                        # publish waypoints not yet completed msg
                        self.waypoints_completed_pub.publish(False)
                    else:
                        # No waypoints received yet
                        self.waypoint_distance_pub.publish(Float64(-1))
                else:                
                    if (not self.random_waypoints):
                        # publish waypoints completed msg:                    
                        # Finished with waypoints from file
                        self.waypoints_completed_pub.publish(True)
                    elif ((rospy.Time.now() - self.last_waypoint_created) > rospy.Duration(secs = 1)):
                        if (self.target_waypoint_id < self.num_waypoints):
                            # generate new waypoint
                            self.last_waypoint_created = rospy.Time.now()
                            p = Point()
                            new_heading = np.random.uniform(self.min_heading, self.max_heading)
                            new_wp_distance = np.random.randint(self.min_distance, self.max_distance)

                            p.x = self.uuv_position[1] + new_wp_distance * math.sin(math.radians(self.uuv_yaw) + new_heading)
                            p.y = self.uuv_position[0] + new_wp_distance * math.cos(math.radians(self.uuv_yaw) + new_heading)
                            p.z = round(self.uuv_position[2])
                            if (self.target_waypoint_id % 2  == 0):
                                p.z = np.random.randint(30,50) #Todo: Sea Depth
                            self.new_wp_pub.publish(p)
                            # print("\n*********************\nnew waypoint: \n" + str(p))
                        else:
                            # publish waypoints completed msg:                    
                            # Finished with random waypoints
                            self.waypoints_completed_pub.publish(True)

    def limit_turnrate(self, limit, heading):    
        return max(-limit, min(limit, heading))

    def dist(self, pos):
        return np.sqrt((self.uuv_position[0] - pos[1])**2 +
                       (self.uuv_position[1] - pos[0])**2)

    def callback_home_position(self, msg):
        self.home_position_msg = msg

    def vector_to_np(self, v):
        return np.array([v.x, v.y, v.z])
    
    def obstacle_avoidance_callback(self, msg):                
        self.hsd_obstacle_avoidance_msg.header.stamp = msg.header.stamp
        self.hsd_obstacle_avoidance_msg.heading = msg.heading
        # self.hsd_obstacle_avoidance_msg.speed = msg.speed
        # self.hsd_obstacle_avoidance_msg.depth = msg.depth

    def callback_next_wp(self, msg):
        if (msg.data):
            self.target_waypoint_id+=1
            rospy.loginfo('--- Next waypoint ---')

    # def callback_obstacle_map(self, msg):
    #     self.map_size = msg.info.width #update the default value, same as height
    #     self.map_origin = msg.info.origin
    #     self.obstacle_map = np.array(msg.data).reshape((msg.info.height, msg.info.width))
    #     # rospy.loginfo('Origin: %0.2f, %0.2f' %(self.map_origin.position.x,self.map_origin.position.y))
       
if __name__=='__main__':
    print('Starting  Task: Waypoint')
    rospy.init_node('task_waypoint', log_level=rospy.INFO)
    try:
        node = TaskWaypoint()
        rospy.spin()
    except rospy.ROSInterruptException:
        print('caught exception')
