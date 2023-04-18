#!/usr/bin/env python

import os
import sys
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
from std_msgs.msg import Float32MultiArray
from scipy.spatial.transform import Rotation as R

class SpawnPipes():
    def __init__(self):
        ''' 
            Randomly spawn a pipeline to the seafloor
        '''
        self.rng = np.random.RandomState(seed=rospy.get_param('~random_seed'))
        self.angle_min = float(rospy.get_param('~angle_min', -0.7854))
        self.angle_max = float(rospy.get_param('~angle_max', 0.7854))
        self.length_min = float(rospy.get_param('~length_min', 50.0))
        self.length_max = float(rospy.get_param('~length_max', 100.0))
        self.pipe_scale = float(rospy.get_param('~pipe_scale', 3.0)) / 5.0 # division for compatibiliy
        self.pipe_posx = -float(rospy.get_param('~pipe_posx', 0.0)) # (-) for compatibiliy
        self.pipe_posy = -float(rospy.get_param('~pipe_posy', 0.0)) # (-) for compatibiliy
        self.ocean_depth = float(rospy.get_param('~ocean_depth', 60.0))
        num_segments = rospy.get_param("~num_segments", 5) 
        package_path     = RosPack().get_path('vandy_bluerov')
        self.pipe_segment    = "pipe_segment"

        self.gazebo_namespace = "/gazebo"

        self.templ_file = os.path.join(
            package_path, 'world_models', 'obstacle', 'pipe_template.urdf'
        )

        self.pipe_markers = []
        # Pipe segments for rviz
        self.pipe_markers_pub = rospy.Publisher(
            '/pipeline/marker', Marker, queue_size=1)
        # Pipe coordinates for post process plots
        self.pipe_plotmarker_pub = rospy.Publisher(
            '/pipeline/plotmarker', Float32MultiArray, queue_size=1)

        # Wait for other nodes to init
        while rospy.Time.now() < rospy.Time(1):
            pass
        
        # Spawn pipe to gazebo & rviz
        pipe_coords = [[self.pipe_posx,self.pipe_posy]]
        for i in range(num_segments):
            self.call_spawn_service(i)
            pipe_coords.append([self.pipe_posx,self.pipe_posy])
        # Publish pipe coords for post process plot
        self.pipe_plotmarker_pub.publish(Float32MultiArray(data = np.array(pipe_coords).flatten()))

    def get_pipe_pose(self,  pipe_length, pipe_angle, x=0, y=0, z=0):
        ''' Create a pose object at x = <pipe_distance>

        Assumes pipe pose is spawned using the vehicle's base link frame
        '''
        pipe_pose = Pose()

        r = R.from_euler('zyx', [pipe_angle, 0, 0], degrees=False)        
        pos = r.apply([0, pipe_length/2, 0])

        pipe_pose.position.x = self.pipe_posx + pos[0]
        pipe_pose.position.y = self.pipe_posy + pos[1]
        pipe_pose.position.z = -self.ocean_depth + self.pipe_scale/2
        self.pipe_posx = pipe_pose.position.x + pos[0]
        self.pipe_posy = pipe_pose.position.y + pos[1]

        pipe_orient = quaternion_from_euler(1.5708, 0, pipe_angle)        
        pipe_pose.orientation.x = pipe_orient[0]
        pipe_pose.orientation.y = pipe_orient[1]
        pipe_pose.orientation.z = pipe_orient[2]
        pipe_pose.orientation.w = pipe_orient[3]
        return pipe_pose

    def call_spawn_service(self, pipe_num, x=0, y=0, z=0):
        ''' Plug coords and model into gazebo spawn service
        '''
        pipe_length = self.rng.uniform(self.length_min, self.length_max)
        pipe_angle = self.rng.uniform(self.angle_min, self.angle_max)                
        pose = self.get_pipe_pose(pipe_length, pipe_angle, x, y, z)

        # Marker is for RVIZ visualization
        marker = Marker()
        
        marker.header.stamp = rospy.Time.now()
        marker.id = len(self.pipe_markers)
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        marker.pose.position = pose.position
        marker.pose.orientation = pose.orientation
        
        marker.header.frame_id = "/world"
        marker.scale.x = self.pipe_scale * 2
        marker.scale.y = self.pipe_scale * 2
        marker.scale.z = pipe_length
        marker.color.b = 0.8
        marker.color.g = 0.1
        marker.color.r = 0.1
        marker.color.a = 1.0
        marker.lifetime = rospy.Duration()
        
        self.pipe_markers.append(marker)
        self.pipe_markers_pub.publish(marker)
        # print(marker)        
        
        # Object for Gazebo (visible by sonar)
        self.spawn_obj( self.pipe_segment + str(pipe_num),
                        self.pipe_segment + str(pipe_num),
                        pose,
                        "world",
                        self.gazebo_namespace,
                        marker.scale)
    

        rospy.loginfo("PIPE SEGMENT GENERATED **************done")
    
    def spawn_obj(self, name, ns, pose, reference_frame, gazebo_ns, scale):
        '''
        Function to spawn object
        '''
        xml = self.get_model(scale.x/2.0, scale.y/2.0, scale.z, name)

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


    def get_model(self, sx, sy, sz, name):
        '''
        Load in the urdf and fill in parameters
        '''
        with open(self.templ_file, "r") as file:
            urdf = file.readlines()
            urdf = ''.join(urdf)
            urdf = urdf.format(
                x  = sx,
                z  = sz,
                n  = name
            )
        return urdf


if __name__ == "__main__":
    rospy.init_node("spawn_obstacles", log_level=rospy.INFO)
    rospy.loginfo("spawn obstacles script started")
    try:
        node = SpawnPipes()
        rospy.spin()
    except rospy.ROSInterruptException:
        print('caught exception')
       
