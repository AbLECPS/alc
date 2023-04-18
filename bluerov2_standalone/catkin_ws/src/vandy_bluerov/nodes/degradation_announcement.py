#!/usr/bin/env python

import rospy
import numpy as np
import math
from std_srvs.srv import Empty
from vandy_bluerov.msg import HSDCommand
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from rospy.numpy_msg import numpy_msg

from uuv_gazebo_ros_plugins_msgs.msg import FloatStamped
from std_msgs.msg import Float64
from std_msgs.msg import Bool, String
from vandy_bluerov.msg import LatLonDepth
from message_filters import ApproximateTimeSynchronizer, Subscriber
import tf.transformations as trans

class DegradationAnnouncement(object):
    def __init__(self):
        self.namespace = rospy.get_namespace().replace('/', '')
        rospy.loginfo('Initialize Task: Degradation announcement for <%s>' % self.namespace)

        self.uuv_degradation_mode = rospy.get_param('~uuv_degradation_mode', 'x')
        self.pub = rospy.Publisher('/uuv0/uuv_degradation_mode', String, queue_size=1)   
       
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():               
            self.pub.publish(str(self.uuv_degradation_mode))            
            rate.sleep()   
       
if __name__=='__main__':
    print('Starting Degradation Announcement')
    # rospy.init_node('task_surface', log_level=rospy.DEBUG)
    rospy.init_node('degradation_announcement', log_level=rospy.INFO)
    try:
        node = DegradationAnnouncement()
        rospy.spin()
    except rospy.ROSInterruptException:
        print('caught exception')
