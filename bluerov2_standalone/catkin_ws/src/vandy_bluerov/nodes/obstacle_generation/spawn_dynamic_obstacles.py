#!/usr/bin/python

import os
import sys
import math
import numpy as np
import navpy
import obstacle_loader as ol
import rospy
from rospkg import RosPack
from gazebo_ros import gazebo_interface
import tf
import tf2_ros
from tf.transformations import quaternion_from_euler
from geometry_msgs.msg import Pose, Vector3Stamped
from nav_msgs.msg import Odometry
from tf2_geometry_msgs import PoseStamped, do_transform_vector3

try:
    sys.path.insert(0, os.path.join(
        os.environ['VIRTUAL_ENV'], 'lib/python2.7/site-packages/'
    ))
except:
    pass


class SpawnObstacles():
    def __init__(self):
        """
        Randomly spawn a box with a constant horizontal velocity in the
        path of the vehicle

        Uses a poisson process with random lambda within a user defined
        interval to generate boxes. This makes the occurance of an obstacle
        event random and therefore more deterministic for the tester.

        The following args must be passed in by launch file:
        box_distance_x     (float ) - Meters. How far ahead of the vehicle do you
                                      want boxes to intercept the UUV?
        box_distance_y     (float ) - Meters. How far to the side of the vehicle do
                                      you want boxes to intercept the UUV?
        box_velocity_x     (float ) - Meters per second. Velocity of Obstacle.
        box_velocity_y     (float ) - Meters per second. Velocity of Obstacle.
        box_size_x         (float ) - Meters. Length of Obstacle on the X axis.
        box_size_y         (float ) - Meters. Length of Obstacle on the Y axis.
        box_size_z         (float ) - Meters. Length of Obstacle on the Z axis.
        avg_uuv_speed      (float ) - Meters per second. Velocity of the UUV.
        box_max_cnt        (float ) - Max number of Obstacles to generate.
        random_seed        (int   ) - set seed used program wide by random number
                                      generators
        lambda_low         (int   ) - Seconds. Low end of interval a random poisson
                                      lambda is chosen
        lambda_high        (int   ) - Seconds. High end of interval a random poisson
                                      lambda is chosen
        pipeline_text_file (string) - The location of the pipeline file used for this run.
        """
        rospy.init_node("spawn_dynamic_obstacles")
        rospy.loginfo("spawn dynamic obstacles script started")

        package_path = RosPack().get_path('vandy_bluerov')
        self.templ_file = os.path.join(
            package_path, 'world_models', 'obstacle', 'obstacle_template.urdf'
        )

        np.random.seed(rospy.get_param('~random_seed'))

        self.box_distance_variance = rospy.get_param("~box_distance_variance", 0) 
        self.box_distance_x = float(rospy.get_param('~box_distance_x', 50.0)) 
        self.box_distance_y = float(rospy.get_param('~box_distance_y', 0.0)) 
        self.box_velocity_x = float(rospy.get_param('~box_velocity_x', 0.0)) 
        self.box_velocity_y = float(rospy.get_param('~box_velocity_y', 0.0))       

        self.sz_x = float(rospy.get_param('~box_size_x', 1.0))
        self.sz_y = float(rospy.get_param('~box_size_y', 1.0))
        self.sz_z = float(rospy.get_param('~box_size_z', 1.0))
        self.sz_x = self.sz_x if self.sz_x > 0 else np.random.uniform(0.5, 10)
        self.sz_y = self.sz_y if self.sz_y > 0 else np.random.uniform(0.5, 10)
        self.sz_z = self.sz_z if self.sz_z > 0 else np.random.uniform(0.5, 10)

        self.max_cnt = float(rospy.get_param('~box_max_cnt', 10))
        self.avg_uuv_speed = float(rospy.get_param('~avg_uuv_speed', 1.5))

        self.wall = rospy.get_param('~wall', False)

        self.time_poisson = np.random.uniform(
            rospy.get_param('~lambda_low'), rospy.get_param('~lambda_high')
        )

        self.robot_namespace = ''
        self.gazebo_namespace = "/gazebo"
        self.reference_frame = self.robot_namespace + '/base_link'

        self.counter = 1
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0

        self.from_file = False

        # Pipe info stuffs
        self.num_segs = 0
        if self.max_cnt == 0:
            pipeline_text_file = rospy.get_param('~pipeline_text_file', "pipeline_input.txt")
            self.v_lat = rospy.get_param('~latitude_veh',0)
            self.v_lon = rospy.get_param('~longitude_veh',0)
            self.r_lat = rospy.get_param('~latitude_ref',0)
            self.r_lon = rospy.get_param('~longitude_ref',0)
            self.pipe_ends, self.pipe_lengths, self.num_segs = self.parse_pipe_txt(pipeline_text_file)
            self.pose = None
            self.odom_sub = rospy.Subscriber('/uuv0/pose_gt_ned', Odometry, self.odom_cb, queue_size=1)
            # Make sure we have some vehicle data before starting
            rospy.wait_for_message('/uuv0/pose_gt_ned', Odometry, timeout=100)
        
        # Wait for other nodes to init
        while rospy.Time.now() < rospy.Time(5):
            pass
        # Check for obstacle filename
        # obstacle_filename = (rospy.get_param('~obstacle_filename', 'obstacles_cp4.yaml'))
        obstacle_filename = (rospy.get_param('~obstacle_filename', ''))
        if len(obstacle_filename):
            self.from_file = True
            # Load obstacles from file
            print('\n\n\n# Load obstacles from file\n\n\n')
            params = ol.ObstacleParams
            loader = ol.ObstacleLoader(obstacle_filename)
            obstacles_from_file = loader.get_obstacles()
            # Spawn obstacles loaded
            for i in range(len(obstacles_from_file)):
                # print('                                                                 >>>>>>>>>>>>>>>>>>> '+str(i))
                self.spawn_obstacle(from_file=True, 
                                   x = obstacles_from_file[i][params.BOX_POS_X], 
                                   y = obstacles_from_file[i][params.BOX_POS_Y], 
                                   z = obstacles_from_file[i][params.BOX_POS_Z], 
                                   vel_x = obstacles_from_file[i][params.BOX_VELOCITY_X],
                                   vel_y = obstacles_from_file[i][params.BOX_VELOCITY_Y],
                                   vel_z = obstacles_from_file[i][params.BOX_VELOCITY_Z],
                                   box_size_x = obstacles_from_file[i][params.BOX_SIZE_X], 
                                   box_size_y = obstacles_from_file[i][params.BOX_SIZE_Y], 
                                   box_size_z = obstacles_from_file[i][params.BOX_SIZE_Z])
                rospy.sleep(0.1)
            self.from_file = False
            self.max_cnt = -1


    def spawn_obstacle(self, from_file=False, x=0, y=0, z=0, vel_x=0, vel_y=0, vel_z=0, box_size_x=0, box_size_y=0, box_size_z=0):
        '''
        Function to spawn obstacles
        '''
        if self.max_cnt != -1:
            self.calculate_transform(from_file, vel_x, vel_y, vel_z)
            pose = self.get_box_pose(from_file, x, y, z)
            xml = self.get_obstacle_model()
            self.reset_namespace()

            # Spawn the obstacle in Gazebo and send the xml to the rospy param
            # server
            name = "box{0}".format(self.counter)
            if (not from_file):
                gazebo_interface.spawn_urdf_model_client(
                    name,
                    xml,
                    self.robot_namespace,
                    pose,
                    self.reference_frame,
                    self.gazebo_namespace
                )
            else:
                gazebo_interface.spawn_urdf_model_client(
                    name,
                    xml,
                    'world',
                    pose,
                    'world',
                    self.gazebo_namespace
                )
            rospy.set_param(name, xml)
            self.counter += 1

            if self.max_cnt == 0:
                self.pipe_ends.pop(0)
                self.pipe_lengths.pop(0)

    def check_finish_condition(self):
        '''
        Function to check if we're done spawning obstacles or not
        '''
        finished = False 
        if (self.max_cnt > 0) and (self.counter >= self.max_cnt):
            finished = True
        elif (self.max_cnt == 0) and ((self.counter-1)  >= self.num_segs):
            finished = True
        elif (self.max_cnt <= 0):
            finished = True

        return finished

    def calc_cur_dis2end(self):
        '''
        Function used to calculate the current distance to the next pipe segment end
        '''
        x2, y2 = self.pipe_ends[0]
        x1, y1 = [self.pose.position.x, self.pose.position.y]
        cur_dis = math.sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2))
        return cur_dis 

    def check_spawn_condition(self, initial=False):
        '''
        Function that checks if we've met the contitions to spawn the next obstacle.
        '''

        if initial and self.max_cnt > 0:
            initial_obstacle_spawn_time = 0.1 + float(rospy.get_param('~dynamic_obstacle_initial_spawn', 0.0))
            rospy.sleep(initial_obstacle_spawn_time)
        elif self.max_cnt > 0:
            rospy.sleep(obstacle_manager.next_spawn())
        elif self.max_cnt == 0:
            while self.calc_cur_dis2end() > self.pipe_lengths[0]/2.0:
                pass
        
    def odom_cb(self, data):
        """
        The callback for the odometry subscriber to grab the current pose of the vehicle.

        Args:
            data (geometery_msgs/pose): The current pose information of the vechicle.
        """
        self.pose = data.pose.pose

    def parse_pipe_txt(self, txt):
        '''
        Loads pipeline info so we can spawn obstacles relative to pipeline segments
        '''

        veh_ned = navpy.lla2ned(self.v_lat, self.v_lon, 0, self.r_lat, self.r_lon, 0)
        pipe_ends = []
        pipe_lengths = []
        f = open(txt, "r")
        for idx,line in enumerate(f):
            tokens = line.split()

            # Get all the ends coordinates
            end = [veh_ned[0] + float(tokens[2]),veh_ned[1] + float(tokens[3])]
            pipe_ends.append(end)

            # Calculate the pipe segment length
            start = [veh_ned[0] + float(tokens[0]),veh_ned[1] + float(tokens[1])]
            p_length = math.sqrt(pow(end[0] - start[0], 2) + pow(end[1] - start[1], 2))
            pipe_lengths.append(p_length)

        return pipe_ends, pipe_lengths, len(pipe_ends)

    def get_obstacle_model(self, from_file=False, size_x=5, size_y=5, size_z=5):
        ''' 
        Load in the obstacle urdf and fill in the velocity and namespace
        '''
        with open(self.templ_file, "r") as file:
            urdf = file.readlines()
            urdf = ''.join(urdf)
            if (not from_file):
                urdf = urdf.format(
                    vx = self.vx, 
                    vy = self.vy, 
                    vz = self.vz, 
                    x  = self.sz_x,
                    y  = self.sz_y,
                    z  = self.sz_z,
                    n  = self.counter
                )
            else:
                urdf = urdf.format(
                    vx = self.vx, 
                    vy = self.vy, 
                    vz = self.vz, 
                    x  = size_x,
                    y  = size_y,
                    z  = size_z,
                    n  = self.counter
                )

        return urdf
    
    def next_spawn(self):
        ''' 
        Get a duration to wait, in seconds, before the next box spawn
        '''
        return np.random.poisson(self.time_poisson)

    def calculate_transform(self, from_file=False, vel_x=0, vel_y=0, vel_z=0):
        ''' 
        Get the transform from uuv0/base_link to world to get the
        obstacle's velocity in the world frame
        
        Assumes the transform from uuv0/base_link to world is being published
        '''
        vec = Vector3Stamped()
        if (not from_file):
            vec.vector.x = self.box_velocity_x + np.random.uniform(-2, 2) * (1 if self.box_distance_variance > 0 else 0)
            vec.vector.y = self.box_velocity_y + np.random.uniform(-2, 2) * (1 if self.box_distance_variance > 0 else 0)
            vec.vector.z = 0
        else:
            vec.vector.x = vel_x
            vec.vector.y = vel_y
            vec.vector.z = vel_z

        print("Dynamic obstacle velocity:")
        print(vec.vector.x)
        print(vec.vector.y)

        tfBuffer = tf2_ros.Buffer(rospy.Duration(10))
        tfListener = tf2_ros.TransformListener(tfBuffer)
        trans = tfBuffer.lookup_transform(
            "world", "uuv0/base_link", rospy.Time(0), rospy.Duration(10)
        )

        vt = do_transform_vector3(vec, trans)

        self.vx = vt.vector.x
        self.vy = vt.vector.y
        self.vz = vt.vector.z

    def get_box_pose(self, from_file=False, x=0, y=0, z=0):
        '''
        Create a pose object at distance
        x = <box_distance_x>, y = <box_distance_y>
        
        Assumes box pose is spawned using the vehicle's base link frame
        '''
        box_pose = Pose()
        if (not from_file):
            box_pose.position.y = self.box_distance_y + np.random.uniform(-self.box_distance_variance, self.box_distance_variance)
            box_pose.position.z = 0.0
            if self.max_cnt > 0:
                box_pose.position.x = self.box_distance_x + np.random.uniform(-self.box_distance_variance, self.box_distance_variance)
            else:
                box_pose.position.x = self.box_distance_x + np.random.uniform(0.0, self.pipe_lengths[0]/8.0)
        else:
            box_pose.position.x = x
            box_pose.position.y = y
            box_pose.position.z = z
        box_pose.orientation.x = 0
        box_pose.orientation.y = 0
        box_pose.orientation.z = 0
        box_pose.orientation.w = 1

        return box_pose

    def reset_namespace(self):
        '''
        Reset the robot namespace to account for the box spawn number
        '''
        self.robot_namespace = "box{0}".format(self.counter)
        self.reference_frame = "uuv0/base_link"

if __name__ == "__main__":
    obstacle_manager = SpawnObstacles()

    obstacle_manager.check_spawn_condition(initial=True)

    while not rospy.is_shutdown():

        # if (obstacle_manager.wall):
        #     for _ in range(10):
        #         obstacle_manager.spawn_obstacle()
        # else:
        
        obstacle_manager.spawn_obstacle()
        if not obstacle_manager.check_finish_condition():
            obstacle_manager.check_spawn_condition()
        else:
            break
