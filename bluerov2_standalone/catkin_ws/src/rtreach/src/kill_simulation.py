#!/usr/bin/python
import rospy

"""very basic node that exits after a set timeout, run in launch a file as a required node, 
which kills the simulation when this node dies"""

if __name__ == "__main__":
    rospy.init_node("kill_simulation")
    args = rospy.myargv()[1:]
    timeout = int(args[0])
    rospy.sleep(2)
    if(timeout<0):
        rospy.spin()
    rospy.sleep(timeout)
    rospy.logwarn("Timeout: Next Experiment")
    rospy.signal_shutdown("Ending current experiment")