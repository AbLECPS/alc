#!/usr/bin/env python

import rospy
import numpy as np
import math
import tf.transformations as trans
import collections
import local_map
import no_go_zones
from scipy.ndimage import measurements as m
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import Range
from geometry_msgs.msg import Pose, Point
from uuv_gazebo_ros_plugins_msgs.msg import FloatStamped
from std_msgs.msg import Float32MultiArray, Float32, Int32
from std_msgs.msg import Header, Bool
from collections import deque
from sensor_msgs.msg import LaserScan

class ObstacleMapping(object):
    def __init__(self):
        self.namespace = rospy.get_namespace().replace('/', '')
        rospy.loginfo('Initialize Obstacle Mapping for <%s>' % self.namespace)
             
        self.size = 1000        
        self.no_go_zone_val = 100
        self.ais_val = 99
        self.obstacle_val = 90
        self.danger_val = 75
        self.empty_val = 0
        self.obstacle_half_size = 10
        # self.track_val = 1

        obstacle_map_msg = OccupancyGrid()
        obstacle_map_msg.header.frame_id = 'uuv0/obstacle_map'        
        obstacle_map_msg.info.resolution = 1
        obstacle_map_msg.info.width = self.size
        obstacle_map_msg.info.height = obstacle_map_msg.info.width
        obstacle_map_msg.info.origin.position.x = - self.size // 2 
        obstacle_map_msg.info.origin.position.y = obstacle_map_msg.info.origin.position.x 
        obstacle_map_msg.data = np.arange(self.size*self.size)
        self.generate_ais_data = rospy.get_param('~generate_ais_data', False)
        predict_ais_data = rospy.get_param('~predict_ais_data', False)
        # self.use_lec3 = rospy.get_param('~use_lec3', False)
        self.obstacle_avoidance_source = rospy.get_param('~obstacle_avoidance_source', "fls_lec3lite")
        seed = rospy.get_param('~seed', 0)
        np.random.seed(seed)

        no_go_zone_list = rospy.get_param('~no_go_zone_list', '')

        self.last_next_wp_pub = rospy.Time.now()

        # Subscriber
        self.target_sub = rospy.Subscriber(
            '/uuv0/target_waypoint', Point, self.callback_target_waypoint, queue_size=1) 
        self.target_waypoint = Point()

        
        self.target_id_sub = rospy.Subscriber(
            '/uuv0/target_waypoint_id', Int32, self.callback_target_waypoint_id, queue_size=1) 
        self.target_waypoint_id = -1

        if self.obstacle_avoidance_source == "fls_lec3lite":                
            # LEC
            self.range_sub = rospy.Subscriber(
                '/lec3lite/ranges', Float32MultiArray, self.callback_fls_lec3lite)
            # AM
            self.am_lec3lite_sub = rospy.Subscriber(
                '/lec3lite/am_vae', Float32MultiArray, self.callback_am_lec3lite)
            self.am_vae_logm = -2
        # BlueROV standard FLS:
        else: #elif self.obstacle_avoidance_source == "fls_echosounder":                
            # FLS echosounder
            self.range_sub = rospy.Subscriber(
                "/uuv0/fls_echosunder", Range, self.callback_range)



        self.fls_contact = 0

        self.doppler_velocity_estimate = np.nan
        self.doppler_distances = deque(maxlen=10)

        self.doppler_velocity_pub = rospy.Publisher(
            "/uuv0/doppler_velocity", Float32, queue_size = 1)

        self.obstacle_in_view_pub = rospy.Publisher(
            "/uuv0/obstacle_in_view", Header, queue_size = 1)

        # Odom/Pose message
        self.odometry_sub = rospy.Subscriber(
             'pose_gt_noisy_ned', Odometry, self.callback_odometry, queue_size=1) 
        self.uuv_position = [0,0,0]
        self.uuv_rpy = [0,0,0]        

        # Publisher
        self.obstacle_near_wp_pub = rospy.Publisher(
            "/uuv0/obstacle_near_wp", Int32, queue_size = 1)

        # Obstacle map
        self.obstacle_map_pub = rospy.Publisher(
            "/uuv0/obstacle_map", OccupancyGrid, queue_size = 1)
        
        self.obstacle_map_occupied_pub = rospy.Publisher(
            "/uuv0/obstacle_map_occupied", Float32MultiArray, queue_size = 1)
        
        self.obstacle_map_update_pub = rospy.Publisher(
            '/uuv0/obstacle_map_update', Bool, queue_size=1)   
            
        self.uuv_heading = 0
                
        # Empty map
        self.map = np.ndarray((self.size, self.size), buffer=np.zeros((self.size, self.size), dtype=np.int), dtype=np.int)
        self.map.fill(0)

        # Empty differential map
        self.map_diff = np.ndarray((self.size, self.size), buffer=np.zeros((self.size, self.size), dtype=np.int), dtype=np.int)
        self.map_diff.fill(0)

        # Check for waypoints close to obstacle
        self.obstacle_avoidance_direction_pub = rospy.Publisher(
            '/uuv0/obstacle_on_waypoint', Bool, queue_size=1) 

        # Random dynamic obstacle positions for AIS
        dynamic_obstacles_count =  rospy.get_param('~dynamic_obstacles_count', 20)
        if self.generate_ais_data:
            # Get AIS update
            self.generate_ais(dynamic_obstacles_count)
            self.ais_distance_pub = rospy.Publisher(
                '/uuv0/ais_distance', Float32MultiArray, queue_size=1) 

        # Define local map:
        # BlueROV FLS range = 30m -> map size: 60x60m
        if self.obstacle_avoidance_source == "fls_echosounder":
            self.fls_range = 30 #m
            self.fls_beam_angle = 30 #deg
        elif self.obstacle_avoidance_source == "fls_lec3lite":
            self.fls_range = 27 #m
            self.fls_beam_angle = 4.5 #deg
        else:
            self.fls_range = 50 #m
            self.fls_beam_angle = 1.7 #deg            

        self.lm = local_map.LocalMap(self.size, self.fls_range * 2, "obstacle_map_local")
        
        if len(no_go_zone_list) > 0:
            self.ngz = no_go_zones.NoGoZones()
            zone_polygons = np.array(no_go_zone_list)  
            for polygon in zone_polygons:            
                self.ngz.set_no_go_zone(self.map, polygon, self.no_go_zone_val)
        else:
            rospy.loginfo('No Go Zone list is empty - skipping zone loader')

        # 1Hz loop
        hz = 20
        rate = rospy.Rate(hz)
        while not rospy.is_shutdown():
            # Update map
            self.map_obstacle(hz)
            
            if self.generate_ais_data:
                # Update AIS data mock
                # 1: Clean old positions
                self.map[self.map == self.ais_val-1] = self.empty_val
                self.map[self.map == self.ais_val] = self.empty_val
            
                # Update ais estimation
                self.update_ais(predict_ais_data, hz)
                # Publish ais - uuv disntances
                self.calculate_ais_distances()
            
            # Get and publish local map
            self.lm.publish_local_map(self.lm.get_local_map(self.map))

            # Publish map
            obstacle_map_msg.header.stamp = rospy.Time.now()
            obstacle_map_msg.data = self.map.flatten()
            self.obstacle_map_occupied_pub.publish(self.get_occupied_cells(self.map))
            self.obstacle_map_pub.publish(obstacle_map_msg)

            rate.sleep()
    
    def get_occupied_cells(self, map):
        msg = Float32MultiArray()
        cells = np.array(np.where(map))
        msg.data = cells.flatten()
        # have to reshape to (2,:) for coords when processing
        return msg

    @staticmethod
    def vector_to_np(v):
        return np.array([v.x, v.y, v.z])

    def odometry_callback(self, msg):
        # Calculate the position, position, and time of message
        p = self.vector_to_np(msg.pose.pose.position)
        self.uuv_position = p

    def dist(self, pos):
        return np.sqrt((self.uuv_position[0] - pos[0])**2 +
                       (self.uuv_position[1] - pos[1])**2)

    def calculate_ais_distances(self):
        ais_distance = []
        for obstacle in self.dynamic_obstacles:
            distance = self.dist([obstacle[0] - self.size//2, obstacle[1]-self.size//2])
            ais_distance.append(distance)
        msg =  Float32MultiArray()
        # msg.data = ais_distance
        msg.data = [np.min(ais_distance)]
        self.ais_distance_pub.publish(msg)
        # print(msg.data)

    def update_ais(self, predict_ais_data, hz):
        for obstacle in self.dynamic_obstacles:
            if  (obstacle[0] < (self.size - self.obstacle_half_size)) and (obstacle[0] > self.obstacle_half_size) and\
                (obstacle[1] < (self.size - self.obstacle_half_size)) and (obstacle[1] > self.obstacle_half_size):

                obstacle[0] = obstacle[0] + obstacle[3] * math.cos(obstacle[2]) * np.sign(obstacle[2] / hz)
                obstacle[1] = obstacle[1] + obstacle[3] * math.sin(obstacle[2]) * np.sign(obstacle[2] / hz)
                if  (obstacle[0] < (self.size - self.obstacle_half_size)) and (obstacle[0] > self.obstacle_half_size) and\
                    (obstacle[1] < (self.size - self.obstacle_half_size)) and (obstacle[1] > self.obstacle_half_size):
                    self.map[int(obstacle[0]), int(obstacle[1])] = self.ais_val                
                    self.update_ais_map(obstacle, self.empty_val, self.ais_val)

                # project movement
                if predict_ais_data:
                    for distance in np.arange(1,50):
                        obstacle_projection = [0,0]
                        obstacle_projection[0] = obstacle[0] + obstacle[3] * (math.cos(obstacle[2]) / hz) * np.sign(obstacle[2]) * distance
                        obstacle_projection[1] = obstacle[1] + obstacle[3] * (math.sin(obstacle[2]) / hz) * np.sign(obstacle[2]) * distance
                        if  (obstacle_projection[0] < (self.size - self.obstacle_half_size)) and (obstacle_projection[0] > self.obstacle_half_size) and\
                            (obstacle_projection[1] < (self.size - self.obstacle_half_size)) and (obstacle_projection[1] > self.obstacle_half_size):
                            # self.map[int(obstacle[0]), int(obstacle[1])] = self.ais_val                
                            self.update_ais_map(obstacle_projection, self.empty_val, self.ais_val-1)

    def generate_ais(self, dynamic_obstacles_count):
        vessel_distance = 210   # [m]
        vessel_speed = 1        # [m/s]

        do_north = np.array(range(50, self.size-50, vessel_distance))                      # Y
        vessel_count = len(do_north)
        do_north = do_north.reshape(vessel_count, 1)
        do_north = np.append(do_north, 300 * np.ones((vessel_count,1)), axis=1)            # X
        do_north = np.append(do_north, np.pi * np.ones((vessel_count,1)), axis=1)        # Direction
        do_north = np.append(do_north, vessel_speed * np.ones((vessel_count,1)), axis=1)   # Speed

        do_south = np.array(range(50, self.size-50, vessel_distance))                      # Y
        do_south = do_south.reshape(vessel_count, 1)
        do_south = np.append(do_south, 400 * np.ones((vessel_count,1)), axis=1)            # X
        do_south = np.append(do_south, 2*np.pi * np.ones((vessel_count,1)), axis=1)          # Direction
        do_south = np.append(do_south, vessel_speed * np.ones((vessel_count,1)), axis=1)   # Speed

        self.dynamic_obstacles = np.append(do_north, do_south, axis=0)
        print("\033[1;34m \n AIS VESSELS \n \033[0m")
        print("\033[1;34m " +str(self.dynamic_obstacles)+ "\033[0m")

    def update_ais_map(self, obstacle, get_val, set_val):
        for i in np.arange(-self.obstacle_half_size, self.obstacle_half_size + 1):
            for j in np.arange(-self.obstacle_half_size, self.obstacle_half_size + 1):
                # bounding box
                if ( 
                    np.abs(i) == self.obstacle_half_size or
                    np.abs(j) == self.obstacle_half_size
                ):
                    if  self.map[int(obstacle[0]+i), int(obstacle[1]+j)] == get_val :
                        self.map[int(obstacle[0]+i), int(obstacle[1]+j)] = set_val

    def map_obstacle(self, hz):
        updated = False
        self.map_diff.fill(0)
        idx=0
        if np.isscalar(self.fls_contact):
            self.fls_contact = [self.fls_contact]
        # print(self.fls_contact)
        for contact in self.fls_contact:
            # If there is a valid reading
            if (contact > 1.5 and contact < self.fls_range):   
                rospy.loginfo_throttle(1, "[Obstacle_mapping] OBSTACLE HIT "+str(contact))
                if idx == 0:
                    for i in np.arange(1, int(contact)):
                        self.set_free_cell(self.uuv_heading, i, hz)     
                updated = self.set_obstacle_cell(self.uuv_heading, contact)
            elif contact > 1.5:            
                for i in np.arange(1, int(contact)+1):
                    updated = self.set_free_cell(self.uuv_heading, i, hz)
            idx+=1
        
        # Publish info
        if self.generate_ais_data:
            updated = True # AIS does auto 1Hz update
        self.obstacle_map_update_pub.publish(Bool(updated))

    def set_free_cell(self, heading, distance, hz):
        theta = heading - math.radians(self.fls_beam_angle/2)
        # Rotation Matrix on Z (yaw) axis
        # somewhat narrower (4 deg)
        updated = False
        range_end = max(3, self.fls_beam_angle-1)
        for j in np.arange(2, range_end):		
            # Get R matrix on Z (yaw) axis
            R = np.array([[np.cos(theta), np.sin(theta)],
                            [-np.sin(theta), np.cos(theta)]])

            # Get contact position
            pos = np.dot(R, np.array([0, distance])) + np.array([self.uuv_position[0] + self.size // 2, self.uuv_position[1] + self.size // 2])

            # Set self.obstacle_val
            #decrese_val = 10 / hz
            if self.obstacle_avoidance_source == "fls_echosounder":
                # BlueROV FLS
                decrese_val = 1
            else:
                # Iver FLS
                decrese_val = 20

            if self.empty_val < self.map[int(pos[0]), int(pos[1])] <= self.obstacle_val:
                # self.map[int(pos[0]), int(pos[1])] = self.empty_val
                self.map[int(pos[0]), int(pos[1])] = max(self.map[int(pos[0]), int(pos[1])] - decrese_val, 0)
                self.map_diff[int(pos[0]), int(pos[1])] = -1
                if self.map[int(pos[0]), int(pos[1])] == self.empty_val:
                    updated = True
            # +1 deg beam
            theta+=math.radians(1)
        return updated

    def set_obstacle_cell(self, heading, distance):
        wp_distance = np.inf
        theta = heading - math.radians(self.fls_beam_angle/2)
        # Rotation Matrix on Z (yaw) axis
        for i in np.arange(0, int(math.ceil(self.fls_beam_angle))):		
            # Get R matrix on Z (yaw) axis
            R = np.array([[np.cos(theta), np.sin(theta)],
                            [-np.sin(theta), np.cos(theta)]])

            # Get contact position
            pos = np.dot(R, np.array([0, distance])) + np.array([self.uuv_position[0] + self.size // 2, self.uuv_position[1] + self.size // 2])

            # Set self.obstacle_val
            self.map[int(pos[0]), int(pos[1])] = self.obstacle_val
            self.map_diff[int(pos[0]), int(pos[1])] = 1
            # Check for target waypoint distance:
            wp_distance = min(self.check_wp([ int(pos[0]) - self.size//2 , int(pos[1]) - self.size//2 ]), wp_distance)                
            # Set self.danger_val ~aura
            '''
            for i in np.arange(-2,3):
                for j in np.arange(-2,3):
                    if  self.map[int(pos[0]+i), int(pos[1]+j)] < (self.empty_val + 1) :
                        self.map[int(pos[0]+i), int(pos[1]+j)] = self.danger_val
                        self.map_diff[int(pos[0]+i), int(pos[1]+j)] = 1
                        # Check for target waypoint distance
                        # wp_distance = min(self.check_wp([int(pos[0]+i) - self.size // 2, int(pos[1]+j) - self.size // 2]), wp_distance)               
            '''
            # +1 deg beam
            theta+=math.radians(1)
        wp_threshold_distance = 30
        if (wp_distance > 0 
            and rospy.Time.now() - self.last_next_wp_pub > rospy.Duration.from_sec(1.0)
            and wp_distance < wp_threshold_distance
        ):
            self.obstacle_near_wp_pub.publish(Int32(self.target_waypoint_id))
            s="\n\n\nObstacle estimated distance to WP: "+str(round(wp_distance,2))+"m\n\n\n"
            rospy.loginfo(s)
            self.last_next_wp_pub  = rospy.Time.now()
        else:
            self.obstacle_near_wp_pub.publish(Int32(-1))
        return True

    def check_wp(self, obstacle_pos):
        # print(str(self.target_waypoint.x) + ' ' + str(obstacle_pos[0]) + '       ' + str(self.target_waypoint.y) + ' ' + str(obstacle_pos[1]))
        return np.sqrt((self.target_waypoint.x - obstacle_pos[1])**2 +
                       (self.target_waypoint.y - obstacle_pos[0])**2)

    def update_obstacle_in_view(self):
        msg = Header()
        msg.stamp = rospy.Time.now()
        self.obstacle_in_view_pub.publish(msg)
    
    def update_doppler(self, fls_distance):
        self.doppler_velocity_estimate = np.nan
        if 0.5 < fls_distance < self.fls_range:
            # if there is a valid reading, update deque
            self.doppler_distances.append(fls_distance)
        else:
            # If no more obstacles reported, clear list
            self.doppler_distances.clear()
        # If there is at least 2 readings
        d_vel = []
        if len(self.doppler_distances) > 1:
            for i in np.arange(len(self.doppler_distances) - 1):
                d_vel.append(self.doppler_distances[i+1] - self.doppler_distances[i])
            self.doppler_velocity_estimate = np.average(d_vel) / 0.02 # ~50Hz for
            # print(self.doppler_velocity_estimate)
            self.update_obstacle_in_view()
        msg = Float32()
        msg.data = self.doppler_velocity_estimate
        self.doppler_velocity_pub.publish(msg)

    def callback_range(self, msg):
        self.fls_contact = msg.range      
        self.update_doppler(self.fls_contact)        
    
    def callback_am_lec3lite(self, msg):
        self.am_vae_logm = msg.data[-2]
        # rospy.logwarn(self.am_vae_logm)
        pass

    def callback_fls_lec3lite(self, msg):
        ranges = np.array(msg.data)
        if self.am_vae_logm < 5:            
            ranges[ranges==np.inf] = -1
            self.fls_contact = ranges
        else:
            rospy.logwarn("lec3_am_vae is high!")
            self.fls_contact = np.zeros_like(ranges)
        # Not applicable for multiple contacts
        # self.update_doppler(self.fls_contact) 

    def callback_odometry(self, msg):
        pos = [msg.pose.pose.position.x,
               msg.pose.pose.position.y,
               msg.pose.pose.position.z]

        quat = [msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z,
                msg.pose.pose.orientation.w]

        # Calculate the position, position, and time of message
        p = self.vector_to_np(msg.pose.pose.position)
        self.uuv_position = p

        # vel = self.vector_to_mag(msg.twist.twist.linear)

        q = self.quaternion_to_np(msg.pose.pose.orientation)
        rpy = trans.euler_from_quaternion(q, axes='sxyz')
        
        self.uuv_rpy[0] = math.degrees(rpy[0])
        self.uuv_rpy[1] = math.degrees(rpy[1])
        self.uuv_rpy[2] = math.degrees(rpy[2])

        self.uuv_heading = math.pi/2 - rpy[2]
        # yaw = math.radians(90 - self.uuv_rpy[2])
    
    def callback_target_waypoint(self, msg):
        self.target_waypoint = msg
        # print(self.target_waypoint)
    
    def callback_target_waypoint_id(self, msg):
        self.target_waypoint_id = msg.data

    def vector_to_np(self, v):
        return np.array([v.x, v.y, v.z])
    
    def quaternion_to_np(self, q):
        return np.array([q.x, q.y, q.z, q.w])

if __name__=='__main__':
    print('Starting Obstacle Mapping')
    rospy.init_node('obstacle_mapping', log_level=rospy.INFO)
    try:
        node = ObstacleMapping()
        rospy.spin()
    except rospy.ROSInterruptException:
        print('caught exception')
