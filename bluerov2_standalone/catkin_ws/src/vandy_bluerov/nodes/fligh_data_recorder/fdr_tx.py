#!/usr/bin/env python

import os
import sys
import rospy
import numpy as np
import math
import tf.transformations as trans

from geometry_msgs.msg import Point, Pose
from rospkg import RosPack
from gazebo_ros import gazebo_interface
from gazebo_msgs.srv import GetModelState
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Float64MultiArray

class FDR(object):
    '''
    Flight Data Recorder class for Cp3 debris detection
    FDR underwater locator beacons are triggered by water immersion.
    Most emit an ultrasonic 10ms pulse at 1Hz at 37.5kHz once per second.
    Simulated as a 1Hz Point publisher, visualized as a 1x1x1m red box. 
    FDR locator node calculates the Rx signal strength based on FDR - UUV distance.

    '''
    def __init__(self):
        self.fdr_pub = rospy.Publisher(
            '/fdr0/ping', Point, queue_size=1)

        self.waypoint_pub = rospy.Subscriber(
            '/uuv0/waypoints', Float64MultiArray, self.callback_waypoint, queue_size=1)   
        self.waypoints = []  
        
        np.random.seed(rospy.get_param('~random_seed'))
        
        x = 0 # np.random.randint(-30, 30)        
        y = 0 # np.random.randint(5, 30)
        z = -rospy.get_param('~ocean_depth') + 1 # Just on the seafloor
        self.point = Point(x,y,z)
        
        self.fdr_marker_pub = rospy.Publisher(
            '/fdr0/marker', Marker, queue_size=1)

        self.fdr_generated = False

        self.gazebo_namespace = "/gazebo"

        package_path = RosPack().get_path('vandy_bluerov')

        self.templ_file = os.path.join(
            package_path, 'world_models', 'obstacle', 'obstacle_template.urdf'
        )

        # Spawn object
        self.marker = Marker()
        
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            if self.fdr_generated == True:
                # Publish marker
                self.fdr_marker_pub.publish(self.marker)
                # Emit ping
                self.ping(self.point)
            rate.sleep()

    def callback_waypoint(self, msg):
        self.waypoints = np.reshape(msg.data,(-1,7))
        
        if (len(self.waypoints) > 0 and not self.fdr_generated):
            min_x = np.min(self.waypoints[:,0])
            max_x = np.max(self.waypoints[:,0])
            min_y = np.min(self.waypoints[:,1])
            max_y = np.max(self.waypoints[:,1])
            self.point.x = np.random.uniform(low=min_x, high=max_x)
            self.point.y = np.random.uniform(low=min_y, high=max_y)
            self.call_spawn_service()
            self.fdr_generated = True

    def ping(self, point):
        self.fdr_pub.publish(point)

    def call_spawn_service(self):
        pose = Pose()
        pose.position = self.point
        
        self.marker.header.frame_id = "/world"
        self.marker.header.stamp = rospy.Time.now()
        # marker.ns = "/fdr0/marker"
        self.marker.id = 0
        self.marker.type = Marker.CUBE
        self.marker.action = Marker.ADD
        
        self.marker.pose.position = pose.position
        self.marker.pose.orientation.w = 1.0
        self.marker.scale.x = 1.0
        self.marker.scale.y = 1.0
        self.marker.scale.z = 1.0
        self.marker.color.b = 0.0
        self.marker.color.g = 0.0
        self.marker.color.r = 1.0
        self.marker.color.a = 1.0
        self.marker.lifetime = rospy.Duration()
        
        # Rescaleable
        self.spawn_fdr(pose)

        print("\033[1;34m FDR GENERATED @ [" + 
            str(self.point.x) + ", " +
            str(self.point.y) + ", " +
            str(self.point.z) + "] " +
            " \033[0m")
        try:
            rospy.wait_for_service('/gazebo/get_model_state')
            box_coordinates=rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
            coordinates=box_coordinates('fdr0','')
            print(["fdr0",coordinates.pose.position.x,coordinates.pose.position.y,coordinates.pose.position.z,coordinates.pose.orientation.x,coordinates.pose.orientation.y,coordinates.pose.orientation.z,coordinates.pose.orientation.w])
            
        except rospy.ServiceException as e:
            rospy.loginfo("OBSTACLE*******Get Model State service call failed:  {0}".format(e))

    def spawn_fdr(self, pose):
        '''
        Function to spawn object
        '''
        xml = self.get_model()

        # Spawn the obstacle in Gazebo and send the xml to the rospy param
        # server
        name = "fdr0"
        reference_frame= "world"
        ns = 'fdr0'
        gazebo_interface.spawn_urdf_model_client(
            name,
            xml,
            ns,
            pose,
            reference_frame,
            self.gazebo_namespace
        )
        rospy.set_param(name, xml)


    def get_model(self):
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
                x  = 1,
                y  = 1,
                z  = 1,
                n  = 0
            )
        return urdf
    
if __name__=='__main__':
    print('Starting FlightDataRecorder: Transmitter')
    rospy.init_node('fdr_transmitter', log_level=rospy.INFO)
    try:
        node = FDR()
        rospy.spin()
    except rospy.ROSInterruptException:
        print('caught exception')