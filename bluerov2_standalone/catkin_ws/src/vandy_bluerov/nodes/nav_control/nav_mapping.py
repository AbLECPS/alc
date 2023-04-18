#!/usr/bin/env python

import rospy
import navpy
import numpy as np
import time
import math
import tf2_ros
import tf, tf.msg
import random
import wavefront2
import heading
import geometry_msgs.msg
import tf.transformations as trans
# import ais
from std_msgs.msg import Float64
from scipy import ndimage as nd
from tf import TransformListener
from nav_msgs.msg import OccupancyGrid, Odometry
from vandy_bluerov.msg import LatLonDepth, HitRay, FollowLECActionResult
# from geometry_msgs.msg import Pose, TransformStamped, Point, Vector3Stamped
from tf2_geometry_msgs import PoseStamped, do_transform_vector3, do_transform_pose
from vandy_bluerov.msg import HSDCommand

class NavOccupancyGrid(object):
    def __init__(self):
        # Setup the default grid parameters
        self.resolution = 1
        self.length = rospy.get_param("~grid_length", 1000)
        # random_seed = rospy.get_param('~seed', 0)
        self.generate_ais_data = rospy.get_param('~generate_ais_data', False)
        predict_ais_data = rospy.get_param('~predict_ais_data', False)
        seed = rospy.get_param('~seed', 0)
        np.random.seed(seed)
        
        # if self.generate_ais_data:
        # self.ais = ais.AISMapping(self.length, False, 20, seed)

        # Setup the subset grid 
        self.grid_msg = OccupancyGrid()
        self.grid_msg.header.frame_id = 'local_map'
        self.grid_msg.info.resolution = self.resolution
        map_size = 300
        self.grid_msg.info.width = map_size
        self.grid_msg.info.height = map_size
        self.grid_msg.data = range(map_size * map_size)
        scale = 3
        look_ahead = 3

        # Setup the pubs and subs
        # For normal use - CP3:
        self.hsd_pipeline_sub = rospy.Subscriber(
            "/uuv0/hsd_to_waypoint", HSDCommand, self.callback_heading)
        self.waypoint_distance_pub = rospy.Subscriber(
            '/uuv0/distance_to_waypoint', Float64, self.callback_wp_distance) 
        self.waypoint_distance = -1

        self.odometry_sub = rospy.Subscriber(
             'pose_gt_noisy_ned', Odometry, self.callback_odometry, queue_size=1) 
        self.uuv_position = [0,0,0]
        self.uuv_rpy = [0,0,0]

        self.obstacle_map_pub = rospy.Subscriber(
            "/uuv0/obstacle_map", OccupancyGrid, self.callback_obstacle_map)

        self.hsd_pub = rospy.Publisher(
            '/uuv0/hsd_ais_avoidance', HSDCommand, queue_size=1)   
        self.hsd_cmd = HSDCommand()
        self.hsd_cmd.heading = 0

        self.pub = rospy.Publisher(
            "/uuv0/local_map", OccupancyGrid, queue_size = 1)
        # self.pub_latlon = rospy.Publisher(
        #     "/map/latlon", LatLonDepth, queue_size = 1)
        # self.pub_transform = rospy.Publisher(
        #     "/map/transform", TransformStamped, queue_size = 1)
        
        self.local_goal = np.array([0,0])
        self.nav_heading = 0
        self.uuv_yaw = 0

        # Setup the origin/center lat and lon
        self.center_lat = rospy.get_param('~world_lat', 38.971203)
        self.center_lon = rospy.get_param('~world_lon', -76.398464)
        origin_latlon = navpy.ned2lla([-(self.length // 2), -(self.length // 2), 0], self.center_lat, self.center_lon, 0)
        self.origin_lat = origin_latlon[0]
        self.origin_lon = origin_latlon[1]
        
        # Setup the timeout counter for old hits
        # self.timeout_counter = 0
        
        # Setup the holder grid
        self.holder_grid = np.ndarray( (self.length, self.length), buffer=np.zeros((self.length, self.length), dtype=np.int), dtype=np.int)
        self.holder_grid.fill(0)
        
        # Setup the coordinate transformation structure
        # self.transform = TransformStamped()

        # occ_grid = NavOccupancyGrid()
        # listener = TransformListener()
        self.init_grid()

        # # For testing olny:
        # random.seed(20)
        # for i in range(25):
        #     self.holder_grid[int(470+i), int(550+i)] = 1
        #     # self.ais.map[int(497+i), int(505+i)] = 1
        # # self.ais.map = nd.binary_dilation(self.ais.map, iterations=5).astype(int)
        # self.holder_grid = nd.binary_dilation(self.holder_grid, iterations=5).astype(int)

        rate = rospy.Rate(1)
        
        # tfBuffer = tf2_ros.Buffer(rospy.Duration(5))
        # tfListener = tf2_ros.TransformListener(tfBuffer)

        self.hc = heading.HeadingCalculator(max_turnrate = 15)
        # noinspection PyInterpreter
        while not rospy.is_shutdown():
            startTime = time.time()
            # TODO: transform this to UUV RPY instead of TF
   
                
            broadcaster = tf2_ros.StaticTransformBroadcaster()
            static_transformStamped = geometry_msgs.msg.TransformStamped()

            static_transformStamped.header.stamp = rospy.Time.now()
            static_transformStamped.header.frame_id = "world"
            static_transformStamped.child_frame_id = "/local_map"

            static_transformStamped.transform.translation.x = self.uuv_position[1]
            static_transformStamped.transform.translation.y = self.uuv_position[0]
            static_transformStamped.transform.translation.z = -self.uuv_position[2]

            quat = tf.transformations.quaternion_from_euler(0,0,1.5708)
            static_transformStamped.transform.rotation.x = quat[0]
            static_transformStamped.transform.rotation.y = quat[1]
            static_transformStamped.transform.rotation.z = quat[2]
            static_transformStamped.transform.rotation.w = quat[3]

            broadcaster.sendTransform(static_transformStamped)
            
            # Set the UUV in the middle of the subset grid
            offset_from_middle = self.grid_msg.info.width // 2 
            self.grid_msg.info.origin.position.x = -offset_from_middle
            self.grid_msg.info.origin.position.y = -offset_from_middle
           
            local_map = self.get_local_map()
            local_goal = (self.local_goal[0]//scale, self.local_goal[1]//scale)
            start = (self.grid_msg.info.height//2//scale, self.grid_msg.info.width//2//scale)
            scaled_map = self.scale_map(local_map, scale)

            if (
                # local_goal != (0,0) and
                local_goal != start
            ):
                pf = wavefront2.PathFinder(scaled_map, start, local_goal)
                path = pf.route_plan()

                if path is not None:
                    for step in path:
                        local_map[step[0] * scale][step[1] * scale] = 100
                    headings = []
                    for step in path[0:10]:
                        headings.append(-self.hc.get_heading(start, step))
                    self.nav_heading = np.average(headings)
            local_map[local_goal[0] * scale][local_goal[1] * scale] = 99
                
            self.grid_msg.data = local_map.flatten()
            # Publish the grid message
            self.pub.publish(self.grid_msg)
            print("Grid message published, Time taken: " + str(time.time()-startTime) + " secs")
            rate.sleep()

    def scale_map(self, local_map, scale):
        return local_map.reshape((len(local_map)//scale, scale, len(local_map)//scale, scale)).max(3).max(1)    

    def get_local_map(self):
        x = int(-self.uuv_position[1]) - self.grid_msg.info.height//2 + self.length//2
        y = int(self.uuv_position[0]) - self.grid_msg.info.width//2 + self.length//2
        # return np.array(self.ais.map[x:x+self.grid_msg.info.width, y:y+self.grid_msg.info.height])
        return np.array(self.holder_grid[x:x+self.grid_msg.info.width, y:y+self.grid_msg.info.height])
        # print("x: " + str(x) + " " + str(x+self.grid_msg.info.width) + ", y: " + str(y) + " " + str(y+self.grid_msg.info.width))

    def vector_to_np(self, v):
        return np.array([v.x, v.y, v.z])
    
    def quaternion_to_np(self, q):
        return np.array([q.x, q.y, q.z, q.w])
    
    def callback_obstacle_map(self, msg):
        self.holder_grid = np.rot90(np.array(msg.data).reshape((msg.info.height, msg.info.width)))
    
    def callback_wp_distance(self, msg):
        self.waypoint_distance = msg.data

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
        q = self.quaternion_to_np(msg.pose.pose.orientation)
        rpy = trans.euler_from_quaternion(q, axes='sxyz')
        
        self.uuv_rpy[0] = math.degrees(rpy[0])
        self.uuv_rpy[1] = math.degrees(rpy[1])
        self.uuv_rpy[2] = math.degrees(rpy[2])

        self.uuv_yaw = self.hc.get_yaw(msg)
        # Startup timeout
        if rospy.Time.now() > rospy.Time(20):
            self.hsd_cmd.heading = self.hc.limit_turnrate(self.hc.get_heading_diff(self.nav_heading, self.uuv_yaw))
        else:
            self.hsd_cmd.heading = 0

    def callback_heading(self, msg):
        self.local_goal =  np.array([ 
            int(self.grid_msg.info.width //2 - max(min(self.waypoint_distance, (self.grid_msg.info.width  - 10) //2), 20) * 
                math.sin(math.radians(self.uuv_yaw + msg.heading))),
            int(self.grid_msg.info.height//2 + max(min(self.waypoint_distance, (self.grid_msg.info.height  - 10) //2), 20) * 
                math.cos(math.radians(self.uuv_yaw + msg.heading)))
                                    ])
        self.hsd_cmd.speed = msg.speed
        self.hsd_cmd.depth = msg.depth
        self.hsd_pub.publish(self.hsd_cmd)
                
    def init_grid(self):
        for i in range(0, self.grid_msg.info.width* self.grid_msg.info.height):
            self.grid_msg.data[i] = int(0)
    
    def update_grid(self, x, y):        
        # if (self.ais.map[int(x), int(y)] >= 0) and (self.ais.map[int(x), int(y)] < 5):
        #     self.ais.map[int(x), int(y)] += 1
        # else:
        #     self.ais.map[int(x), int(y)] = int(1)
        if (self.holder_grid[int(x), int(y)] >= 0) and (self.holder_grid[int(x), int(y)] < 5):
            self.holder_grid[int(x), int(y)] += 1
        else:
            self.holder_grid[int(x), int(y)] = int(1)
             
if __name__ == '__main__':
    print('Starting Nav Map node')
    rospy.init_node('nav_map_node', log_level=rospy.INFO)
    try:
        node = NavOccupancyGrid()
        rospy.spin()
    except rospy.ROSInterruptException:
        print('caught exception') 

