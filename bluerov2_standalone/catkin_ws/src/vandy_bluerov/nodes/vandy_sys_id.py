#!/usr/bin/env python

# need to subscribe to the steering message and angle message
from message_filters import ApproximateTimeSynchronizer, Subscriber
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import numpy as np 
import os
import rospkg
import atexit
import rospy
from rospkg import RosPack

from vandy_bluerov.msg import HSDCommand


class genSysIDData:
    ''' This class gather's data that will be used to perform greybox system identification of the F1Tenth Racecar in Matlab. 
        The states of this model are [x,y,yaw] where 
            x: x position 
            y: y position 
            yaw: orientation (heading)
        It has two inputs: heading, speed: 
            heading: heading command from uuv 
            speed: speed command from uuv
    '''

    def __init__(self,path):
        r = rospkg.RosPack() 
        # The data will be stored in a csv file in the csv directory
        self.save_path_root=path
        self.odometry_sub=rospy.Subscriber("uuv0/pose_gt", Odometry,self.odom_callback,queue_size = 20)
        self.hsd =rospy.Subscriber("uuv0/hsd_command",HSDCommand,self.master_callback,queue_size = 20)
        #self.sub = ApproximateTimeSynchronizer([self.odometry_sub,self.hsd], queue_size = 20, slop = 0.049)
        self.odom_msg = None
        #self.campaign = 0
        self.filename=self.save_path_root+'{}_{}.csv'.format("sys_id","data")
        self.file = open(self.filename, 'w+')

    def odom_callback(self,msg):
        self.odom_msg = msg
    #callback for the synchronized messages
    def master_callback(self,hsd_msg): 
        
        odom_msg = self.odom_msg
        if(odom_msg):
            # position 
            x = odom_msg.pose.pose.position.x
            y = odom_msg.pose.pose.position.y

            qx = odom_msg.pose.pose.orientation.x
            qy = odom_msg.pose.pose.orientation.y
            qz = odom_msg.pose.pose.orientation.z
            qw = odom_msg.pose.pose.orientation.w

            # Convert Quaternion to rpy
            rpy = euler_from_quaternion([qx,
                                        qy,
                                        qz,
                                        qw])

            # linear velocity 
            velx = odom_msg.twist.twist.linear.x
            vely = odom_msg.twist.twist.linear.y
            velz = odom_msg.twist.twist.linear.z



            # magnitude of velocity 
            speed = np.asarray([velx,vely])
            speed = np.linalg.norm(speed)

        

            # heading 
            heading = hsd_msg.heading

            # throttle 

            hsd_speed = hsd_msg.speed
        
            print("x:",x,"y:",y,"theta:",rpy[2],"speed:",speed,"heading:",heading,"hsd_speed:",hsd_speed)
            self.file.write('%f, %f, %f, %f, %f, %f,%f,%f,%f,%f\n' % (x,y,rpy[2],speed,heading,hsd_speed,qx,qy,qz,qw))

    def shutdown(self):
        self.file.close()
        print('Goodbye')
        

if __name__ == '__main__':
    rospy.init_node('sys_id')
    args = rospy.myargv()[1:]
    path=args[0]
    C = genSysIDData(path)
    atexit.register(C.shutdown)
    r = rospy.Rate(80)

    while not rospy.is_shutdown():
        r.sleep()
    C.shutdown()
