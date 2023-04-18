#!/usr/bin/env python
import os
import rospy
import numpy as np
import math
import csv
import tensorflow as tf
import tensorflow.keras as keras

from std_msgs.msg import Bool
from scipy.optimize import curve_fit
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from geometry_msgs.msg import Point, Pose
from nav_msgs.msg import Odometry
from scipy.interpolate import griddata

class FDRLocator(object):
    '''
    FDR Locator class using CSV FDR Rx data
    '''
    def __init__(self):
        pass
        self.log_filename = rospy.get_param('~log_filename', 'fdr_rx_map.csv')
        self.waypoints_completed =  False
        self.waypoints_completed_sub = rospy.Subscriber(
            '/uuv0/waypoints_completed', Bool, self.callback_waypoints_completed)

        self.fdr_location_pub = rospy.Publisher(
            '/uuv0/fdr_pos_est', Point, queue_size=1)  
        self.fdr_located = False


        # # Tensorflow config
        # tf_config = tf.ConfigProto(log_device_placement=False)
        # tf_config.gpu_options.allow_growth = True
        # # Keep track of TF Session and Graph in case LibraryAdapter instance is called across multiple threads
        # self.session = tf.Session(config=tf_config)
        # self.tf_graph = tf.get_default_graph()
        # keras.backend.set_session(self.session)

    def callback_waypoints_completed(self, msg):
        if (not self.waypoints_completed and 
            msg.data and 
            not self.fdr_located
        ):
            self.process_log()
        self.waypoints_completed = msg.data

    def process_log(self):
        if os.path.isdir(results_dir):
            data = np.genfromtxt(os.path.join(results_dir, self.log_filename), delimiter=',', skip_header=1, comments="#")
            if len(data) > 0:
                x=data[:,[0]].flatten()
                y=data[:,[1]].flatten()
                z=data[:,[2]].flatten() # Noisy raw data
                # z=data[:,[3]].flatten() # GT data
                # z=data[:,[4]].flatten() # LEC filtered data

                grid_x, grid_y = np.mgrid[min(x):max(x):100j, min(y):max(y):100j]
                points = np.array([x,y]).T
                grid_z1 = gaussian_filter(griddata(points, z, (grid_x, grid_y), method='linear', fill_value=0), sigma=10) #nearest/linear
                # grid_z1 = griddata(points, z, (grid_x, grid_y), method='nearest')#nearest/linear

                [fdr_x, fdr_y] = np.unravel_index(np.nanargmax(grid_z1),(len(grid_x),len(grid_y)))
                
                # import matplotlib.pyplot as plt
                # print([fdr_x, fdr_y])
                # plt.imshow(grid_z1.T, extent=(0,1,0,1), origin='lower')
                # plt.plot(fdr_x/len(grid_x), fdr_y/len(grid_y), 'k.', ms=10)
                # plt.title('FDR')

                step_x = grid_x[1,0] - grid_x[0,0]
                step_y = grid_y[0,1] - grid_y[0,0]
                fdr_x = min(x) + step_x * fdr_x
                fdr_y = min(y) + step_y * fdr_y
                print("\033[1;34m FDR Calculated position: [" + 
                    str(fdr_x) + ", " +
                    str(fdr_y) + "] " +
                    " \033[0m")
                p = Point()
                p.x = fdr_x
                p.y =  fdr_y
                # p.z = seabed
                self.fdr_location_pub.publish(p)
                self.fdr_located = True     
            else:
                rospy.loginfo('[FDR Locator] No data in log')
  
if __name__=='__main__':
    print('Starting FlightDataRecorder: Locator')
    rospy.init_node('fdr_locator', log_level=rospy.INFO)
    results_dir = rospy.get_param("~results_directory")
    try:
        node = FDRLocator()
        rospy.spin()
    except rospy.ROSInterruptException:
        print('caught exception')  
