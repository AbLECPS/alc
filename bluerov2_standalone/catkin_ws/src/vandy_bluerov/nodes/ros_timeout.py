#!/usr/bin/env python
import rospy
import os


class ROSTimeout(object):
    """
    ROS node which intentionally dies if the system is still running after a specified timeout has elapsed.
    This is a workaround since ROS seems to have no concept of cleanly shutting down a ROS system.
    Node must be marked as "required" so that ROS master will initiate shutdown of all other nodes when this node dies.
    """
    def __init__(self):
        self._timeout = rospy.get_param('~timeout', None)
        if (self._timeout is None) or (self._timeout <= 0):
            rospy.loginfo('ROSTimeout initialized with no timeout.')
        else:
            rospy.loginfo('ROSTimeout initialized with a timeout of %s seconds.' % str(self._timeout))

            # Assume timeout is provided in seconds and configure a Timer
            rospy.Timer(rospy.Duration(self._timeout), self.shutdown_node, oneshot=True)

    def shutdown_node(self, event):
        rospy.loginfo("ROSTimeout node reached specified timeout of %s seconds" % str(self._timeout))
        rospy.signal_shutdown("ROSTimeout node reached specified timeout of %s seconds" % str(self._timeout))


if __name__ == '__main__':
    print('Starting ROS Timeout Controller')
    rospy.init_node('ros_timeout')
    try:
        node = ROSTimeout()
        rospy.spin()
    except rospy.ROSInterruptException:
        print('caught exception')
