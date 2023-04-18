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
        self.odometry_sub=rospy.Subscriber("uuv0/pose_gt_ned", Odometry,self.master_callback,queue_size = 20)
        self.hsd =rospy.Subscriber("uuv0/delta_hsd",HSDCommand,self.hsd_callback,queue_size = 20)
        
        #self.sub = ApproximateTimeSynchronizer([self.odometry_sub,self.hsd], queue_size = 20, slop = 0.049)
        self.odom_msg = None
        self.hsd_msg = None
        #self.campaign = 0
        self.filename=self.save_path_root+'{}_{}_{}.csv'.format("data","sys_id","odom")
        self.file = open(self.filename, 'w+')


    def hsd_callback(self,msg):
        self.hsd_msg = msg

    #callback for the synchronized messages
    def master_callback(self,odom_msg): 
        
        hsd_msg = self.hsd_msg
        if(hsd_msg):
            # position 
            x = odom_msg.pose.pose.position.x
            y = odom_msg.pose.pose.position.y

            # Convert Quaternion to rpy
            rpy = euler_from_quaternion([odom_msg.pose.pose.orientation.x,
                                        odom_msg.pose.pose.orientation.y,
                                        odom_msg.pose.pose.orientation.z,
                                        odom_msg.pose.pose.orientation.w])

            # linear velocity 
            velx = odom_msg.twist.twist.linear.x
            vely = odom_msg.twist.twist.linear.y
            velz = odom_msg.twist.twist.linear.z



            # magnitude of velocity 
            speed = np.asarray([velx,vely])
            speed = np.linalg.norm(speed)

        

            # heading 
            heading = hsd_msg.heading

            # if(heading>np.pi):

            #     heading = (heading * (180/np.pi)) - 360

            # else:

            #     heading = heading * (180/np.pi)

            # throttle 

            hsd_speed = hsd_msg.speed
        
            print("x:",x,"y:",y,"speed:",speed,"theta:",rpy[2],"heading:",heading,"hsd_speed:",hsd_speed)
            self.file.write('%f, %f, %f, %f, %f, %f\n' % (x,y,rpy[2],speed,heading,hsd_speed))

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