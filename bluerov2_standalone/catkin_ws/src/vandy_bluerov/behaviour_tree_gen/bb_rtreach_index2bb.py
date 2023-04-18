#!/usr/bin/env python
#
# License: BSD
#   https://raw.github.com/stonier/py_trees_ros/license/LICENSE
#
##############################################################################
# Documentation
##############################################################################

"""
This node captures the rtreachability unsafe hyperrectangle index scaled to 0.0 to 1.0 range
"""

##############################################################################
# Imports
##############################################################################

import py_trees
import rospy
import sensor_msgs.msg as sensor_msgs
from std_msgs.msg import Float32
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
        * rtreach_index: the raw message from topic /unsafe_hyperrectangle_index
    Args:
        name (:obj:`str`): name of the behaviour
        topic_name (:obj:`str`) : name of the input topic        
    """
    def __init__(self, 
                    name, 
                    topic_name="rtreach_index"        
                ):
                
        super(ToBlackboard, self).__init__(name=name,
                                           topic_name=topic_name,
                                           topic_type=Float32,
                                           blackboard_variables={"rtreach_index":None},
                                           clearing_policy=py_trees.common.ClearingPolicy.NEVER
                                           )
        self.blackboard = py_trees.blackboard.Blackboard()
        
        self.blackboard.rtreach_index = Float32()
        
        
############<<USER INIT CODE BEGINS>>##############################
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
############<<USER UPDATE CODE ENDS>>################################
        return status
        
############<<USER CUSTOM CODE BEGINS>>##############################
############<<USER CUSTOM CODE ENDS>>################################