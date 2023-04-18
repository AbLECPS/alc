#!/usr/bin/env python
import numpy as np
from sensor_msgs.msg import Image
import sys

class CvBridge(object):
    def __init__(self):
        pass

    def imgmsg_to_cv2(self, img_msg, channels=1):
        '''
        https://answers.ros.org/question/350904/cv_bridge-throws-boost-import-error-in-python-3-and-ros-melodic/
            
            Provides conversions between OpenCV and ROS image formats in a hard-coded way.  
            CV_Bridge, the module usually responsible for doing this, is not compatible with Python 3,
            the language this all is written in.  So we create this module, and all is... well, all is not well,
            but all works.  :-/
        '''
        # if img_msg.encoding != "bgr8":
        #     rospy.logerr("This Coral detect node has been hardcoded to the 'bgr8' encoding.  Come change the code if you're actually trying to implement a new camera")
        dtype = np.dtype("uint8") # Hardcode to 8 bits...
        dtype = dtype.newbyteorder('>' if img_msg.is_bigendian else '<')

        image_opencv = np.ndarray(shape=(img_msg.height, img_msg.width, channels), # and three channels of data. Since OpenCV works with bgr natively, we don't need to reorder the channels.
                        dtype=dtype)
        # If the byt order is different between the message and the system.
        if img_msg.is_bigendian == (sys.byteorder == 'little'):
            image_opencv = image_opencv.byteswap().newbyteorder()
        return image_opencv
    
    def cv2_to_imgmsg(self, cv_image, channels=1):
        '''
        https://answers.ros.org/question/350904/cv_bridge-throws-boost-import-error-in-python-3-and-ros-melodic/
            
            Provides conversions between OpenCV and ROS image formats in a hard-coded way.  
            CV_Bridge, the module usually responsible for doing this, is not compatible with Python 3,
            the language this all is written in.  So we create this module, and all is... well, all is not well,
            but all works.  :-/
        '''
        img_msg = Image()
        img_msg.height = cv_image.shape[0]
        img_msg.width = cv_image.shape[1]
        if channels == 1:
            img_msg.encoding="mono8"
        else:
            img_msg.encoding = "bgr8"
        img_msg.is_bigendian = 0
        img_msg.data = cv_image.tobytes()
        img_msg.step = len(img_msg.data) // img_msg.height 
        return img_msg
       
             
