#!/usr/bin/env python

# Description: Non lec avoidance using obstacle map (occupancy grid) message


import rospy
import numpy as np
import math
import tf.transformations as trans

from std_srvs.srv import Empty
# from vandy_bluerov.msg import Obstacle
from vandy_bluerov.msg import HSDCommand
from nav_msgs.msg import Odometry
from uuv_gazebo_ros_plugins_msgs.msg import FloatStamped
from std_msgs.msg import Float64, String
from message_filters import ApproximateTimeSynchronizer, Subscriber
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point
from sensor_msgs.msg import Range
from nav_msgs.msg import OccupancyGrid


class ObstacleAvoidance(object):
    def __init__(self):
        self.namespace = rospy.get_namespace().replace('/', '')
        rospy.loginfo('Initialize obstacle avoidance for <%s>' % self.namespace)
        
        self.fls_contact = 0
        # # Subscribe FLS obstacle info
        # self.fls_obstacle_sub = rospy.Subscriber(
        #     '/uuv0/fls_obstacle', Obstacle, self.callback_laser_range)

        # FLS echosounder
        self.range_sub = rospy.Subscriber(
            "/uuv0/fls_echosunder", Range, self.callback_range)
        
        self.obstacle_map_sub = rospy.Subscriber(
            "/uuv0/obstacle_map", OccupancyGrid, self.callback_obstacle_map)
        
        self.obstacle_map = np.array([], dtype=np.int)

        # HSD Sub to pipe tracking
        self.hsd_pipeline_mapping_sub= rospy.Subscriber(
            "/uuv0/hsd_pipeline_mapping", HSDCommand, self.callback_hsd_pipe_tracking)
        # HSD sub to RTH 
        self.hsd_rth_sub= rospy.Subscriber(
            "/uuv0/hsd_to_rth", HSDCommand, self.callback_hsd_rth)
        # HSD sub to waypoint 
        self.hsd_waypoint_sub= rospy.Subscriber(
            "/uuv0/hsd_to_waypoint", HSDCommand, self.callback_hsd_waypoint)

        # CM hsd input msg
        self.cm_hsd_input_sub= rospy.Subscriber(
            "/uuv0/cm_hsd_input", String, self.callback_cm_hsd_input)
        self.cm_hsd_input =  String()

        self.hsd_input = HSDCommand()

        self.map_size = 1000 #default, update it from msg
        self.map_origin = Pose()

        self.odom_sub = rospy.Subscriber(
            "/uuv0/pose_gt_noisy", Odometry, self.callback_pose)
        self.uuv_yaw = 0 #rad
        self.uuv_position = Point()

        # Pipe distance    
        self.pipeline_distance_sub = rospy.Subscriber(
            "/uuv0/pipeline_distance_from_mapping", FloatStamped, self.callback_distance)

        self.obstacle_avoidance_direction_sub = rospy.Subscriber(
            '/uuv0/obstacle_avoidance_direction', Float64, self.callback_obstacle_avoidance_direction)
        self.obstacle_avoidance_direction = 1


        self.pipeline_distance = 0

        self.obstacle_safe_distance_threshold = 25# start making HSD to avoid obstacles closer to this
        self.obstacle_critical_distance_threshold = 5 # maximum HSD to avoid obstacles closer to this 

        self.hsd_pub = rospy.Publisher(
            '/uuv0/hsd_obstacle_avoidance', HSDCommand, queue_size=1)   
        self.init_speed = rospy.get_param('~init_speed', 2.0)
        self.init_depth = rospy.get_param('~init_depth', 45)

        self.avoidance_angle_step = rospy.get_param('~avoidance_angle_step', 20)
        self.avoidance_angle_step_ais = rospy.get_param('~avoidance_angle_step_ais', 45)

        self.obstacle_state_pub = rospy.Publisher(
            '/uuv0/fls_obstacle_avoidance_score', Float64, queue_size=1)   
        
        self.hsd_obstacle_avoidance_msg = HSDCommand()
        self.fls_obstacle_avoidance_score_msg = Float64()

        self.empty_val = 0

        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            self.hsd_obstacle_avoidance_msg.speed = self.init_speed
            # rospy.loginfo('gt pose ned yaw %0.1f, yaw+pipe %0.1f' %(self.uuv_yaw, self.uuv_yaw + self.hsd_pipe_tracking.heading ))
            # Check if there is Obstacle ahead            
            if self.cm_hsd_input.data == 'waypoint_task':
                self.hsd_obstacle_avoidance_msg.heading = self.get_safe_heading(self.uuv_yaw, np.sign(self.obstacle_avoidance_direction))            
            else:
                self.hsd_obstacle_avoidance_msg.heading = self.get_safe_heading(self.uuv_yaw, np.sign(self.pipeline_distance))
            # Check if there is Obstacle in the pipe tracking heading (if no obstacle direclty ahead)            
            if self.hsd_obstacle_avoidance_msg.heading == 0.0:
                mission_heading = self.uuv_yaw - self.hsd_input.heading
                if self.cm_hsd_input.data == 'waypoint_task':            
                    mission_avoidance = self.get_safe_heading(mission_heading, np.sign(self.obstacle_avoidance_direction))
                else:
                    mission_avoidance = self.get_safe_heading(mission_heading, np.sign(self.pipeline_distance))
                #if self.hsd_obstacle_avoidance_msg.heading > 0.0:
                if mission_avoidance > 0.0:
                    self.hsd_obstacle_avoidance_msg.heading = self.uuv_yaw - mission_heading + mission_avoidance
                    rospy.logdebug('[Obstacle Avoidance] mission_heading >> %0.1f' %self.hsd_obstacle_avoidance_msg.heading)
                    # self.hsd_obstacle_avoidance_msg.speed = 0.0 
                # rospy.logdebug('Pipetracking/Avoidance: %0.1f  |  %0.1f' %(self.hsd_input.heading, self.hsd_obstacle_avoidance_msg.heading))            
            else:
                # rospy.logdebug('Obstacle ahead, Avoidance:     |  %0.1f' %( self.hsd_obstacle_avoidance_msg.heading))            
                rospy.logdebug('[Obstacle Avoidance] FLS contact >> %0.1f' %self.hsd_obstacle_avoidance_msg.heading) 
                # self.hsd_obstacle_avoidance_msg.speed = 0.0

            self.hsd_obstacle_avoidance_msg.header.stamp = rospy.Time.now() 
            self.fls_obstacle_avoidance_score_msg.data= (-1)
            self.hsd_obstacle_avoidance_msg.depth = self.init_depth
            # self.hsd_obstacle_avoidance_msg.speed = self.init_speed
            self.hsd_pub.publish(self.hsd_obstacle_avoidance_msg)
            self.obstacle_state_pub.publish(self.fls_obstacle_avoidance_score_msg)
            
            rate.sleep()

    def check_obstacle_free_hdg(self, position, heading):
        # Check route for obstacles
        # in 0 to FLS max range
        # rospy.logdebug(heading)
        ret = 0
        distance = 50
        # panic under min disance
        # todo
        heading = np.radians(heading)
        # route=[]
        # if map initaliezed
        # rospy.logdebug('      %d %d' %(position.x + self.map_size // 2, position.y + self.map_size // 2))

        if self.obstacle_map.size > 0:
            for i in range(1, distance):	
                # Get R matrix on Z (yaw) axis
                R = np.array([[np.cos(heading), -np.sin(heading)],
                                [np.sin(heading), np.cos(heading)]])
                # Get contact position
                pos = np.dot(R, np.array([i,0])) + np.array([position.x + self.map_size // 2, position.y + self.map_size // 2])
                # rospy.logdebug( self.obstacle_map[int(pos[0), int(pos[1])] )
                # route.append(self.obstacle_map[int(pos[1]), int(pos[0])])
                # rospy.logdebug('%d %d' %(int(pos[1]), int(pos[0])))
                if self.obstacle_map[int(pos[1]), int(pos[0])] > self.empty_val:
                    ret = self.obstacle_map[int(pos[1]), int(pos[0])]           
                    # rospy.loginfo('   !  Obstacle: %0.1f'%np.degrees(heading))      
                    break                
        # rospy.logdebug(route)      
        return ret
                
    def get_safe_heading(self, heading, side):        
        # Side: 
        #     1 right
        #    -1 left
        # Heading: 
        #    global heading
        heading_mod = 0
        if side == 0:
            side=1
        self.hsd_obstacle_avoidance_msg.heading = 0 # deg, relative, 0: no change 
        self.hsd_obstacle_avoidance_msg.speed = 0 # m/s, relative, 0: no change
        self.hsd_obstacle_avoidance_msg.depth = 0 # m, relative, 0: no change

        # theta = heading
        # Scan obstacles 
        for i in range(0, 9):	
            obstacle = self.check_obstacle_free_hdg(self.uuv_position, heading + heading_mod)
            if obstacle == 0:   
                if i > 0:
                    heading_mod += self.avoidance_angle_step * side
                break
            # Angle step
            if obstacle > 90:
                return self.avoidance_angle_step_ais * side        
            else:
                heading_mod += self.avoidance_angle_step * side
            rospy.logdebug(heading_mod)
        if heading_mod == 0:
            pass
            # rospy.logdebug('Safe')
        return -heading_mod
        
    def callback_range(self, msg):
        self.fls_contact = msg.range

    def callback_laser_range(self, msg):
        self.fls_contact = min(msg.ranges)
    
    def callback_obstacle_map(self, msg):
        self.map_size = msg.info.width #update the default value, same as height
        self.map_origin = msg.info.origin
        self.obstacle_map = np.array(msg.data).reshape((msg.info.height, msg.info.width))
        # rospy.loginfo('Origin: %0.2f, %0.2f' %(self.map_origin.position.x,self.map_origin.position.y))

    def callback_pose(self, odom):
        # Convert Quaternion to rpy
        rpy = trans.euler_from_quaternion([odom.pose.pose.orientation.x,
                                     odom.pose.pose.orientation.y,
                                     odom.pose.pose.orientation.z,
                                     odom.pose.pose.orientation.w])        
        self.uuv_position = odom.pose.pose.position
        self.uuv_yaw = np.degrees(rpy[2])
        # rospy.loginfo('UUV: %0.2f, %0.2f, %0.2f' %(self.uuv_position.x,self.uuv_position.y, math.degrees(self.uuv_yaw)))

    def callback_cm_hsd_input(self, msg):
        self.cm_hsd_input = msg

    def callback_hsd_pipe_tracking(self, msg):
        if self.cm_hsd_input.data == 'tracking_task':
            self.hsd_input = msg

    def callback_hsd_rth(self, msg):
        if self.cm_hsd_input.data == 'rth_task':            
            self.hsd_input = msg  

    def callback_hsd_waypoint(self, msg):
        if self.cm_hsd_input.data == 'waypoint_task':            
            self.hsd_input = msg   

    def callback_distance(self, msg):
        self.pipeline_distance = msg.data

    def callback_obstacle_avoidance_direction(self, msg):
        self.obstacle_avoidance_direction = msg.data

if __name__=='__main__':
    print('Starting obstacle avoidance')
    # rospy.init_node('obstacle_avoidance', log_level=rospy.DEBUG)
    rospy.init_node('obstacle_avoidance', log_level=rospy.INFO)
    try:
        node = ObstacleAvoidance()
        rospy.spin()
    except rospy.ROSInterruptException:
        print('caught exception')
