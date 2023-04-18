#!/usr/bin/env python
#
# License: BSD
#   https://raw.github.com/stonier/py_trees_ros/license/LICENSE
#
##############################################################################
# Documentation
##############################################################################

"""
This node captures the rtreachabiliy decision result over safety
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
import numpy as np
import math
from collections import deque
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
        * rtreach_result: the raw message from topic /reachability_result
        * emergency_stop_warning (:obj:`bool`)
        * rtreach_warning (:obj:`bool`)
        * rtreach_long_term_warning (:obj:`bool`)
    Args:
        name (:obj:`str`): name of the behaviour
        topic_name (:obj:`str`) : name of the input topic        
        enable_emergency_stop (:obj:`float`) : parameter        
        rtreach_window_size (:obj:`float`) : parameter        
        rtreach_window_threshold (:obj:`float`) : parameter        
    """
    def __init__(self, 
                    name, 
                    topic_name="rtreach_result",         
                    enable_emergency_stop=True,         
                    rtreach_window_size=25,         
                    rtreach_window_threshold=0.75        
                ):
                
        super(ToBlackboard, self).__init__(name=name,
                                           topic_name=topic_name,
                                           topic_type=Float32,
                                           blackboard_variables={"rtreach_result":None},
                                           clearing_policy=py_trees.common.ClearingPolicy.NEVER
                                           )
        self.blackboard = py_trees.blackboard.Blackboard()
        
        self.blackboard.rtreach_result = Float32()
        
        self.blackboard.emergency_stop_warning = False
        self.blackboard.rtreach_warning = False
        self.blackboard.rtreach_long_term_warning = False
        
        self.enable_emergency_stop=enable_emergency_stop        
        self.rtreach_window_size=rtreach_window_size        
        self.rtreach_window_threshold=rtreach_window_threshold        
############<<USER INIT CODE BEGINS>>##############################
        self.rtreach_window = deque(maxlen=rtreach_window_size)
        self.rtreach_long_term_pub = rospy.Publisher( '/uuv0/rtreach_long_term',
                                            Float32,
                                            queue_size=1)   
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
        # Old way, using binary rtreach output
        # if self.blackboard.rtreach_result.data < 1.0 and self.enable_emergency_stop:
        #     self.blackboard.emergency_stop_warning = True
        #     rospy.logwarn_throttle(1, "%s: emergency_stop_warning!" % self.name)
        
        # long term rtreach
        if rospy.Time.now() > rospy.Time(5):            
            val = (max(
                        max(
                            min(math.exp(self.blackboard.rtreach_index.data) / 4.0, 0.5)
                            ,0),
                        self.blackboard.rtreach_result.data)
                    )
            self.rtreach_window.append(val)
            # 1: safe, 0: unsafe
            if (np.mean(self.rtreach_window) < self.rtreach_window_threshold) and len(self.rtreach_window) == self.rtreach_window_size:
                self.blackboard.rtreach_long_term_warning = True
                rospy.logwarn("%s: **** rtreach_long_term_warning (%0.2f)" % (self.name, val))
            else:
                self.blackboard.rtreach_long_term_warning = False

            if self.blackboard.rtreach_result.data < 1.0:
                self.blackboard.rtreach_warning = True
                rospy.logwarn("%s: rtreach_warning" % self.name)
            else:
                self.blackboard.rtreach_warning = False
            self.rtreach_long_term_pub.publish(Float32(np.mean(self.rtreach_window)))


############<<USER UPDATE CODE ENDS>>################################
        return status
        
############<<USER CUSTOM CODE BEGINS>>##############################
############<<USER CUSTOM CODE ENDS>>################################