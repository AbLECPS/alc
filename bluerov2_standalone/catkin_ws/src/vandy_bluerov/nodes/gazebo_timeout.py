#!/usr/bin/env python
import rospy
import os
import time


class GazeboTimeout(object):
    """
    ROS node which intentionally dies if Gazebo fails to start properly.
    Node must be marked as "required" so that ROS master will initiate shutdown of all other nodes when this node dies.
    """
    def __init__(self):
        timeout = rospy.get_param('~timeout', 30)
        ros_time = rospy.Time.now()
        init_time = time.time()
        while not rospy.is_shutdown():
            if time.time() > init_time + timeout:
                if rospy.Time.now() == ros_time:
                    rospy.logerr("GazeboTimeout node reached specified timeout of %s seconds" % str(timeout))
                    if os.path.isdir(results_dir):            
                        with open(os.path.join(results_dir, "_GAZEBO_TIMEOUT_"), 'w') as fd:
                            rospy.logerr(os.path.join(results_dir, "_GAZEBO_TIMEOUT_"))
                            fd.close()
                    rospy.signal_shutdown('')
            time.sleep(1)   

if __name__ == '__main__':
    print('Starting Gazebo Timeout Controller')
    rospy.init_node('gazebo_timeout')
    results_dir = rospy.get_param("~results_directory")

    try:
        node = GazeboTimeout()
        rospy.spin()
    except rospy.ROSInterruptException:
        print('caught exception')
