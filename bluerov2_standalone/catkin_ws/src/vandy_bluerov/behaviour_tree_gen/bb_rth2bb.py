#!/usr/bin/env python
#
# License: BSD
#   https://raw.github.com/stonier/py_trees_ros/license/LICENSE
#
##############################################################################
# Documentation
##############################################################################

"""
This node captures if the uuv is commanded to return to home
"""

##############################################################################
# Imports
##############################################################################

import py_trees
import rospy
import sensor_msgs.msg as sensor_msgs
from std_msgs.msg import Bool
from py_trees_ros import subscribers
############<<USER IMPORT CODE BEGINS>>##############################
############<<USER IMPORT CODE ENDS>>################################

##############################################################################
# Blackboard node
##############################################################################


class ToBlackboard(subscribers.ToBlackboard):
    """
    Subscribes to the battery message and writes battery data to the blackboard.
    Also adds a warning flag to the blackboard if the battery
    is low - note that it does some buffering against ping-pong problems so the warning
    doesn't trigger on/off rapidly when close to the threshold.

    When ticking, updates with :attr:`~py_trees.common.Status.RUNNING` if it got no data,
    :attr:`~py_trees.common.Status.SUCCESS` otherwise.

    Blackboard Variables:
        * bb_rth: the raw message from topic /uuv0/bb_rth
        * bb_rth_warning (:obj:`bool`)
    Args:
        name (:obj:`str`): name of the behaviour
        topic_name (:obj:`str`) : name of the input topic        
        failsafe_rth_enable (:obj:`float`) : parameter        
    """
    def __init__(self, 
                    name, 
                    topic_name="bb_rth",         
                    failsafe_rth_enable=True        
                ):
                
        super(ToBlackboard, self).__init__(name=name,
                                           topic_name=topic_name,
                                           topic_type=Bool,
                                           blackboard_variables={"bb_rth":None},
                                           clearing_policy=py_trees.common.ClearingPolicy.NEVER
                                           )
        self.blackboard = py_trees.blackboard.Blackboard()
        
        self.blackboard.bb_rth = Bool()
        
        self.blackboard.bb_rth_warning = False
        
        self.failsafe_rth_enable=failsafe_rth_enable        
############<<USER INIT CODE BEGINS>>##############################
        self.blackboard.bb_rth.data = False
############<<USER INIT CODE ENDS>>################################
    def update(self):
        """
        Call the parent to write the raw data to the blackboard and then check against the
        parameters to update the bb variable
        """
        self.logger.debug("%s.update()" % self.__class__.__name__)
        status = super(ToBlackboard, self).update()
        if status == py_trees.common.Status.RUNNING:
            return status
############<<USER UPDATE CODE BEGINS>>##############################
        if (self.blackboard.bb_rth.data and self.failsafe_rth_enable):
            self.blackboard.bb_rth_warning = True
            rospy.logwarn_throttle(1, "%s: RTH triggered!" % self.name)
        # else:
        #     self.blackboard.bb_rth_warning = False
        # else don't do anything in between - i.e. avoid the ping pong problems

        self.feedback_message = "bb_rth triggered" if self.blackboard.bb_rth_warning else "bb_rth not triggered"
############<<USER UPDATE CODE ENDS>>################################
        return status
        
############<<USER CUSTOM CODE BEGINS>>##############################
############<<USER CUSTOM CODE ENDS>>################################