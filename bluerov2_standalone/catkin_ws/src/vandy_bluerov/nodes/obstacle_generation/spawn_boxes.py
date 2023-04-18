#!/usr/bin/env python

import os
import sys
import obstacle_loader as ol
try:
    sys.path.insert(0, os.path.join(os.environ['VIRTUAL_ENV'], 'lib/python2.7/site-packages/'))
except:
    pass
import numpy as np
from tf.transformations import quaternion_from_euler

import rospy
from rospkg import RosPack
from gazebo_ros import gazebo_interface
from gazebo_msgs.srv import GetModelState
import csv
from geometry_msgs.msg import Pose
from visualization_msgs.msg import Marker, MarkerArray


class SpawnBoxes():
    def __init__(self):
        ''' Randomly spawn a box in the path of the vehicle

        Uses a poisson process with random lambda within a user defined interval to generate
        boxes. This makes the occurrence of an obstacle event random and therefore more
        deterministic for the tester.

        The following args must be passed in by launch file:
        random_seed (int) - set seed used program wide by random number generators
        lambda_low (int) - Seconds. Low end of interval a random poisson lambda is chosen
        lambda_high (int) - Seconds. High end of interval a random poisson lambda is chosen
        sdf (bool) - box mesh config. sdf if true else urdf
        box_distance (int) - Meters. How far from the vehicle do you want boxes to spawn?
        box_size_x         (float ) - Meters. Length of Obstacle on the X axis.
                                      If 0 then random from 0.5 to 5m
        box_size_y         (float ) - Meters. Length of Obstacle on the Y axis.
                                      If 0 then random from 0.5 to 5m
        box_size_z         (float ) - Meters. Length of Obstacle on the Z axis.
                                      If 0 then random from 0.5 to 5m


        Optional arguments:
            box_distance (int) - Meters. How far from the vehicle do you want boxes to spawn?
            debris - ???
            initial_spawn_time (float) - Seconds. Initial time to spawn an obstacle.
            fixed_spawn_interval (float) - Seconds. Time to wait between spawning obstacles. Overrides lambda value.
        '''
        np.random.seed(rospy.get_param('~random_seed'))
        self.poisson_lambda = np.random.uniform(rospy.get_param('~lambda_low'),
                                                rospy.get_param('~lambda_high'))
        rospy.loginfo('Poisson lambda used: {}'.format(self.poisson_lambda))
        self.box_size_x = float(rospy.get_param('~box_size_x', 1.0))
        self.box_size_y = float(rospy.get_param('~box_size_y', 1.0))
        self.box_size_z = float(rospy.get_param('~box_size_z', 1.0))
        self.wall = rospy.get_param('~enable_wall', True)

        package_path     = RosPack().get_path('vandy_bluerov')
        self.box_name    = "static_box"

        # self.robot_namespace  = rospy.get_namespace().replace('/', '')
        self.robot_namespace  = 'uuv0'
        self.gazebo_namespace = "/gazebo"

        self.templ_file = os.path.join(
            package_path, 'world_models', 'obstacle', 'obstacle_template.urdf'
        )

        self.reference_frame  = self.robot_namespace + '/base_link'
        self.box_markers = []
        self.box_markers_pub = rospy.Publisher('~collision_objects', Marker, queue_size=1)

        # add field for obstacle positions
        self.box_locations = [['Box_Name', 'x', 'y', 'z', 'quat_x', 'quat_y', 'quat_z', 'quat_w']]

        # Optional parameters
        self.box_distance = rospy.get_param("~box_distance", 60)  # spawn distance from vehicle
        self.box_distance_variance = rospy.get_param("~box_distance_variance", 0) 
        self.debris = rospy.get_param("~enable_debris", False)
        self.initial_spawn_time = rospy.get_param("~initial_spawn_time", -1.0)
        self.fixed_spawn_interval = rospy.get_param("~fixed_spawn_interval", -1.0)
        self.random = rospy.get_param("~enable_random", False)

        # Misc variables
        self._first_obstacle_spawned = False

        # Wait for other nodes to init
        while rospy.Time.now() < rospy.Time(3):
            pass
        # Check for obstacle filename
        # obstacle_filename = (rospy.get_param('~obstacle_filename', 'obstacles_cp4.yaml'))
        obstacle_filename = (rospy.get_param('~obstacle_filename', ''))
        if len(obstacle_filename):
            # Load obstacles from file
            print('\n\n\n# Load obstacles from file\n\n\n')
            params = ol.ObstacleParams
            loader = ol.ObstacleLoader(obstacle_filename)
            obstacles_from_file = loader.get_obstacles()
            # Spawn obstacles loaded
            for i in range(len(obstacles_from_file)):
                # print('                                                                 >>>>>>>>>>>>>>>>>>> '+str(i))
                self.call_spawn_service(
                    i,
                    from_file=True,
                    x = obstacles_from_file[i][params.BOX_POS_X],
                    y = obstacles_from_file[i][params.BOX_POS_Y],
                    z = obstacles_from_file[i][params.BOX_POS_Z],
                    box_size_x = obstacles_from_file[i][params.BOX_SIZE_X],
                    box_size_y = obstacles_from_file[i][params.BOX_SIZE_Y],
                    box_size_z = obstacles_from_file[i][params.BOX_SIZE_Z])
                rospy.sleep(0.1)
    
    def next_spawn(self):
        ''' Get a duration to wait, in seconds, before the next box spawn
        '''
        if self._first_obstacle_spawned is False and self.initial_spawn_time >= 0:
            return max(self.initial_spawn_time - rospy.get_time(), 0)

        else:
            if self.fixed_spawn_interval >= 0:
                return self.fixed_spawn_interval
            else:
                return np.random.poisson(self.poisson_lambda)

    def get_box_pose(self, from_file=False, x=0, y=0, z=0):
        ''' Create a pose object at x = <box_distance>

        Assumes box pose is spawned using the vehicle's base link frame
        '''

        box_pose = Pose()
        if not from_file:
            box_pose.position.x = self.box_distance + np.random.uniform(-self.box_distance_variance, self.box_distance_variance)
            # box_pose.position.z += self.box_size_y / 2 # Pose correction
            box_pose.position.z = 0 # Pose correction
            # VU mod
            # box_pose.position.y = np.random.uniform(-15,15)
            # generate orientation parallel to ground, between 0-90 degrees
            if (not self.wall):
                box_orient = quaternion_from_euler(0, 0, np.random.uniform(0, 1.5708))
            else:
                box_orient = quaternion_from_euler(0, 0, 0)
            if (self.debris):
                # box_pose.position.x = 30
                box_pose.position.x = np.random.uniform(15,30)
                box_pose.position.z = -13 # Todo: seafloor!
                #box_pose.position.y = -30
                box_pose.position.y = np.random.uniform(-30,30)
                box_orient = quaternion_from_euler(0, 0, 20)
            elif (self.random):
                box_pose.position.z = 0
                box_pose.position.y = np.random.uniform(-self.box_distance_variance, self.box_distance_variance)
                box_orient = quaternion_from_euler(np.random.uniform(0, 1.5708), np.random.uniform(0, 1.5708), np.random.uniform(0, 1.5708))
            box_pose.orientation.x = box_orient[0]
            box_pose.orientation.y = box_orient[1]
            box_pose.orientation.z = box_orient[2]
            box_pose.orientation.w = box_orient[3]
        else:
            box_pose.position.x = x
            box_pose.position.y = y
            box_pose.position.z = z
            box_orient = quaternion_from_euler(0, 0, np.random.uniform(0, 1.5708))
            box_pose.orientation.x = box_orient[0]
            box_pose.orientation.y = box_orient[1]
            box_pose.orientation.z = box_orient[2]
            box_pose.orientation.w = box_orient[3]
        return box_pose

    def call_spawn_service(self, box_num, from_file=False, x=0, y=0, z=0, box_size_x=0, box_size_y=0, box_size_z=0):
        ''' Plug coords and model into gazebo spawn service

        Randomly chooses one of 5 box models, assigns it a name, and spawns it.
        '''
        # if (self.debris):
        #     box_model_num = np.random.randint(4, 5)
        # else:
        #    box_model_num = np.random.randint(1, 5)box_model_num = 2 # 1x1 m box to scale
        box_model_num = 2  # 1x1 m box to scale

        pose = self.get_box_pose(from_file, x, y, z)

        # Marker is for RVIZ visualization
        marker = Marker()
        
        marker.header.stamp = rospy.Time.now()
        marker.ns = "/uuv0/collision_objects"
        marker.id = len(self.box_markers)
        #marker.type = Marker.MESH_RESOURCE #mesh resource not scalable - why?
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose.position = pose.position
        marker.pose.orientation = pose.orientation
        if not from_file:
            marker.header.frame_id = self.reference_frame
            marker.scale.x = self.box_size_x if self.box_size_x > 0 else np.random.uniform(0.5, 10)
            marker.scale.y = self.box_size_y if self.box_size_y > 0 else np.random.uniform(0.5, 10)
            marker.scale.z = self.box_size_z if self.box_size_z > 0 else np.random.uniform(0.5, 10)
        else:
            marker.header.frame_id = "/world"
            marker.scale.x = box_size_x# if box_size_x > 0 else np.random.uniform(0.5, 10)
            marker.scale.y = box_size_y# if box_size_x > 0 else np.random.uniform(0.5, 10)
            marker.scale.z = box_size_z# if box_size_x > 0 else np.random.uniform(0.5, 10)
        marker.color.b = 0.5
        marker.color.g = 0.5
        marker.color.r = 0.5
        marker.color.a = 1.0
        marker.lifetime = rospy.Duration()
        self.box_markers.append(marker)
        self.box_markers_pub.publish(marker)
        
        # Object for Gazebo (visible by sonar)
        if not from_file:
            self.spawn_obj( self.box_name + str(box_num),
                            self.robot_namespace,
                            pose,
                            self.reference_frame,
                            self.gazebo_namespace,
                            marker.scale)
        else:
            # self.spawn_obj( self.box_name + '_from_file_' + str(box_num),
            self.spawn_obj( self.box_name + str(box_num),
                            self.box_name + str(box_num),
                            pose,
                            "world",
                            self.gazebo_namespace,
                            marker.scale)
        self._first_obstacle_spawned = True

        rospy.loginfo("OBSTACLE GENERATED **************done")
        try:
            rospy.wait_for_service('/gazebo/get_model_state')
            box_coordinates=rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
            # if not from_file:
            coordinates=box_coordinates(self.box_name + str(box_num),'')
            # else:
            #     coordinates=box_coordinates(self.box_name + '_from_file_' + str(box_num),'')
            self.box_locations.append([self.box_name + str(box_num),coordinates.pose.position.x,coordinates.pose.position.y,coordinates.pose.position.z,coordinates.pose.orientation.x,coordinates.pose.orientation.y,coordinates.pose.orientation.z,coordinates.pose.orientation.w])
            rospy.logdebug(self.box_locations)
        except rospy.ServiceException as e:
            rospy.loginfo("OBSTACLE*******Get Model State service call failed:  {0}".format(e))

    def spawn_obj(self, name, ns, pose, reference_frame, gazebo_ns, scale):
        '''
        Function to spawn object
        '''
        xml = self.get_model(scale.x, scale.y, scale.z)

        # Spawn the obstacle in Gazebo and send the xml to the rospy param

        gazebo_interface.spawn_urdf_model_client(
            name,
            xml,
            ns,
            pose,
            reference_frame,
            gazebo_ns
        )
        rospy.set_param(name, xml)


    def get_model(self, sx, sy, sz):
        '''
        Load in the urdf and fill in the velocity and namespace
        '''
        with open(self.templ_file, "r") as file:
            urdf = file.readlines()
            urdf = ''.join(urdf)
            urdf = urdf.format(
                vx = 0,
                vy = 0,
                vz = 0,
                x  = sx,
                y  = sy,
                z  = sz,
                n  = 0
            )
        return urdf


if __name__ == "__main__":
    rospy.init_node("spawn_obstacles")
    rospy.loginfo("spawn obstacles script started")
    box_count = rospy.get_param("~num_obstacles", -1)
    random_box_density = rospy.get_param("~random_box_density", 1)
    timeout = rospy.get_param("~timeout", -1)
    sb = SpawnBoxes()
    num_spawned = 0
    if sb.debris:
        rospy.sleep(5)
    else:
        rospy.sleep(timeout)
    while not rospy.is_shutdown() and ((box_count==-1) or (num_spawned<box_count)):
        rospy.sleep(sb.next_spawn())
        print(">>>>>>>>>>>>>>>>>> BOX COUNT : " + str(box_count))
        if (sb.debris or sb.random):
            if sb.debris:
                objects = 5
            else:
                objects = random_box_density
            for _ in range(objects):
                sb.call_spawn_service(num_spawned)
                num_spawned += 1
        else:
            sb.call_spawn_service(num_spawned)
            num_spawned += 1

