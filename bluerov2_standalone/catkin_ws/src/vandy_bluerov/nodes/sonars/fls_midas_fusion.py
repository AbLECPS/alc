#!/usr/bin/env python



import rospy
import numpy as np
import math
import scipy.misc
import cv2
from image_geometry import PinholeCameraModel
from sensor_msgs.msg import Range, Image, CameraInfo
from cv_bridge import CvBridge
from std_msgs.msg import Header, Float32MultiArray
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from collections import deque
from sensor_msgs.msg import PointCloud
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Point32
import open3d as o3d


'''
pip install -U open3d-python
'''

class FLSMiDaSFusion(object):
    def __init__(self):     
        np.set_printoptions(suppress=True)
        self.fls_contact = 0
        self.avg_center_val = 1
        self.rgb_image = None
        self.obstacle_avoidance_source = rospy.get_param('~obstacle_avoidance_source', "fls_pencilbeam")
        if self.obstacle_avoidance_source == "fls_echosounder":
            self.fls_range = 30 #m
            self.fls_beam_angle = 30 #deg
        else:
            self.fls_range = 50 #m
            self.fls_beam_angle = 1.7 #deg            
        self.filter_size = 3
        self.cv_buffer = deque(maxlen=self.filter_size)
        # self.pcm = PinholeCameraModel()
        self.cvbridge = CvBridge()

        self.pointcloud_pub = rospy.Publisher(
            '/midas_pointcloud', PointCloud, queue_size=1)

        # Subscribe to MiDaS depth map
        self.midas_sub = rospy.Subscriber(
            '/midas_topic', Image, self.callback_midas, queue_size=1)    

        # # Subscribe to camera info
        # self.camera_info_sub = rospy.Subscriber(
        #     '/uuv0/camera/camera_info', CameraInfo, self.callback_camera_info, queue_size=1)    
        self.rgb_image = []
        self.camera_info_sub = rospy.Subscriber(
             '/uuv0/camera/image_raw', Image, self.callback_camera, queue_size=1)    

        self.range_sub = rospy.Subscriber(
            "/uuv0/fls_echosunder", Range, self.callback_range)       
        
        self.midas_absolute_pub = rospy.Publisher(
            '/uuv0/midas_absolute', Image, queue_size=1)   
        
        range_limit=50 #m
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            if len(self.cv_buffer) == self.filter_size and self.rgb_image is not None:
                # Filter depths to remove some fluctuation
                depth_raw = np.reshape(self.fls_contact * np.mean(self.cv_buffer, axis=0) / self.avg_center_val, (256,256))
                # Limit depth
                depth_raw[depth_raw>range_limit * 2] = range_limit * 2  
                # Deblur image
                depth_raw = gaussian_filter(depth_raw, 1)              
                
                # draw filled circles in white on black background as masks
                mask = np.zeros_like(depth_raw)
                mask = cv2.circle(mask, (128,128), 128, (255,255,255), -1)

                # put mask into alpha channel of input
                depth_raw = cv2.cvtColor(depth_raw, cv2.COLOR_GRAY2RGBA)
                depth_raw[:, :, 3] = mask[:,:]
                # depth_raw = cv2.cvtColor(depth_raw, cv2.COLOR_RGBA2GRAY)
                
                # Publish the depth map
                self.midas_absolute_pub.publish(self.cvbridge.cv2_to_imgmsg(
                    cv2.cvtColor(depth_raw, cv2.COLOR_RGBA2GRAY),
                    encoding="passthrough")
                )

                cv2.imwrite("depth_raw.png", depth_raw)
                o3d_depth = o3d.io.read_image("depth_raw.png")
                # o3d_rgb = o3d.io.read_image("rgb_raw.png")

                
                rgbd_image = o3d.geometry.create_rgbd_image_from_color_and_depth(o3d_depth, o3d_depth)

                # K: [127.99952983061029, 0.0, 128.5, 0.0, 127.99952983061029, 128.5, 0.0, 0.0, 1.0]
                # fx = 127.99952983061029
                # fy = 127.99952983061029 
                # cx = 128.5
                # cy = 128.5
                # Intrinsic camera matrix for the raw (distorted) images.
                #     [fx  0 cx]
                # K = [ 0 fy cy]
                #     [ 0  0  1]
                # Projects 3D points in the camera coordinate frame to 2D pixel
                # coordinates using the focal lengths (fx, fy) and principal point
                # (cx, cy).

                pcd = o3d.geometry.create_point_cloud_from_depth_image(
                    rgbd_image.depth, 
                    o3d.camera.PinholeCameraIntrinsic(
                        256,256,127.99952983061029,127.99952983061029,128.5,128.5
                    )
                )
                # print(pcd)
                pcd = o3d.geometry.voxel_down_sample(pcd,voxel_size=0.00001)
  
                cl, ind = o3d.geometry.radius_outlier_removal(pcd,
                                                  nb_points=64,
                                                  radius=5)
                

                hfov = np.pi/2
                vfov = np.pi/2

                rot = {
                    'r': np.pi,
                    'p': np.pi/2,
                    'y': np.pi/2
                }
                pcd = pcd.rotate([ rot['r'], rot['p'], rot['y'] ], center=False)

                pc = PointCloud()            
                header = Header()
                header.stamp = rospy.Time.now()
                header.frame_id = 'uuv0/camera_link_depth'
                pc.header = header
                # scale is not right at loading
                pcd = np.asarray(pcd.points)*250000
                pcd2 = []
                for p in pcd:
                    # min/max ranges
                    if 10 < p[0] < 55:
                        pcd2.append(p)
                pcd = pcd2
                for p in (pcd):
                    pc.points.append(Point32(p[0],p[1],p[2]))                    
                self.pointcloud_pub.publish(pc)
            rate.sleep()   

    def callback_camera(self, msg):
        self.rgb_image = self.cvbridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        # cv2.imwrite("rgb_raw.png", self.rgb_image)


    # def callback_camera_info(self, msg):
    #     self.camera_params = self.pcm.fromCameraInfo(msg)
    #     print(self.camera_params)
        
    def callback_range(self, msg):
        self.fls_contact = msg.range      

    def callback_fls_lec3(self, msg):
        if msg.data != float("inf"):
            self.fls_contact = msg.data
        else:
            self.fls_contact = -1

    def callback_laser_range(self, msg):
        self.fls_contact = self.fls_range if abs(min(msg.ranges))==np.inf else min(msg.ranges)

    def callback_midas(self, msg):

        # Get dept map
        depth_raw = self.cvbridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        # Fuse depth map with FLS distance
        kernel = 32
        reduced_img = cv2.resize(depth_raw, (kernel, kernel), interpolation = cv2.INTER_NEAREST)
        # Create ROI
        k = self.gaussian_kernel(kernel, 16)
        k[k>0.75] = 1
        k[k<=0.75] = 0
        # Select center region
        gaussian_ranges = reduced_img * k        
        # Get max pixel value for scaling range
        self.avg_center_val = np.max(gaussian_ranges[k == 1]) / 255
        # Add absolute depth map to buffer
        self.cv_buffer.append([(1 - (depth_raw/255))])
        # Create output from buffer
        

    def gaussian_kernel(self, n=256, sigma=16):       
        x = np.zeros((101, 101))
        x[50, 50] = 1
        g = gaussian_filter(x, sigma)
        g = g / np.max(g)
        return cv2.resize(g, (n, n), interpolation = cv2.INTER_CUBIC)
       
if __name__=='__main__':
    print('Starting Task: FLS_MiDaS_Fusion')
    # rospy.init_node('task_surface', log_level=rospy.DEBUG)
    rospy.init_node('FLS_MiDaS_Fusion', log_level=rospy.INFO)
    try:
        node = FLSMiDaSFusion()
        rospy.spin()
    except rospy.ROSInterruptException:
        print('caught exception')
