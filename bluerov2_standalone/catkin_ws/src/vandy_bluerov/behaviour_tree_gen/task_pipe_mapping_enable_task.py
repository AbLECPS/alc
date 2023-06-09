#!/usr/bin/env python
#
# License: BSD
#   https://raw.github.com/stonier/py_trees_ros/license/LICENSE
#
##############################################################################
# Documentation
##############################################################################

##############################################################################
# Imports
##############################################################################

import dynamic_reconfigure.client
import py_trees
import rospy            

 
from std_msgs.msg import Bool

############<<USER IMPORT CODE BEGINS>>##############################
############<<USER IMPORT CODE ENDS>>################################

##############################################################################
# Behaviours
##############################################################################


class TaskHandler(py_trees.behaviour.Behaviour):
    
    def __init__(self, 
                 name  
                ):                 
        super(TaskHandler, self).__init__(name=name)
        self.task = name
        self.blackboard = py_trees.blackboard.Blackboard()
        self.blackboard.pipe_mapping_enable = Bool()
     

############<<USER INIT CODE BEGINS>>##############################
        self.blackboard.pipe_mapping_enable = True

        self.bt_use_lec2_left_pub = rospy.Publisher(
            "/bt/use_lec2/left", Bool, queue_size = 1)
        self.bt_use_lec2_right_pub = rospy.Publisher(
            "/bt/use_lec2/right", Bool, queue_size = 1)
############<<USER INIT CODE ENDS>>################################

    def setup(self, timeout):
############<<USER SETUP CODE BEGINS>>##############################
        self.feedback_message = "setup"
############<<USER SETUP CODE ENDS>>################################
        return True            



    def update(self):
        """
        Call the parent to write the raw data to the blackboard and then check against the
        parameters to update the bb variable
        """
        self.logger.debug("%s.update()" % self.__class__.__name__)
        
############<<USER UPDATE CODE BEGINS>>##############################
        self.blackboard.pipe_mapping_enable = True
        self.bt_use_lec2_left_pub.publish(Bool(not self.blackboard.lec2_am_l_pipe_warning))
        self.bt_use_lec2_right_pub.publish(Bool(not self.blackboard.lec2_am_r_pipe_warning))
############<<USER UPDATE CODE ENDS>>################################

         # Return always running                
        return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        """
        Shoot off a clearing command.

        Args:
            new_status (:class:`~py_trees.common.Status`): the behaviour is transitioning to this new status
        """
############<<USER TERMINATE CODE BEGINS>>##############################
        self.feedback_message = "cleared"
############<<USER TERMINATE CODE ENDS>>################################        

############<<USER CUSTOM CODE BEGINS>>##############################
############<<USER CUSTOM CODE ENDS>>################################