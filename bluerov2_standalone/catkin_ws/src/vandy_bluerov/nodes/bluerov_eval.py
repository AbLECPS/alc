#!/usr/bin/env python
import rospy
import os
import collections
import numpy as np
import math

from uuv_gazebo_ros_plugins_msgs.msg import FloatStamped
from vandy_bluerov.msg import HSDCommand
from std_msgs.msg import Header, Float64, Bool, Float32MultiArray
from nav_msgs.msg import Odometry

class BluerovEval(object):
    def __init__(self):
        rospy.loginfo('\n\033[1;32mStarting BlueROV evaulation \033[0m \n')
        
        # Subscribe to odometry
        self.odometry_sub = rospy.Subscriber(
            '/uuv0/pose_gt_noisy_ned', Odometry, self.callback_pose, queue_size=1)  

        # Pipe distance    
        self.pipeline_distance_sub = rospy.Subscriber(
            "/uuv0/pipeline_distance_from_mapping", FloatStamped, self.callback_pipeline_distance_sub)
        pipeline_distance_msg = FloatStamped()

        # Pipe in view    
        self.pipeline_in_view_sub = rospy.Subscriber(
            "/uuv0/pipeline_in_view", Header, self.callback_pipeline_in_view_sub)
        self.pipeline_in_view_msg = Header()

        # Pipe in view GT
        self.pipeline_in_view_gt_pub = rospy.Subscriber(
            "/uuv0/pipeline_in_view_gt", Header,  self.callback_pipeline_in_view_gt_sub)
        self.pipeline_in_view_gt_msg = Header()
            
        # Pipe pos in SLS    
        self.pipeline_in_sls_sub = rospy.Subscriber(
            "/uuv0/pipeline_in_sls", FloatStamped, self.callback_pipeline_in_sls_sub)
        self.pipeline_in_sls = 0.0

        # Pipe pos in GT   
        self.pipeline_in_gt_sub = rospy.Subscriber(
            "/uuv0/pipeline_in_gt", FloatStamped, self.callback_pipeline_in_gt_sub)
        self.pipeline_in_gt = 0.0

        # Subscribe to collision avoidance
        self.hsd_obstacle_avoidance_sub = rospy.Subscriber(
            '/uuv0/hsd_obstacle_avoidance', HSDCommand, self.callback_obstacle_avoidance)

        # Subscribe to degradation GT
        self.degradation_gt_sub = rospy.Subscriber(
            '/uuv0/degradation_gt', Float32MultiArray, self.callback_degradation_gt) 
        #Starting Time, Thruster or Nominal, Efficiency
        self.degradation_gt = [-1 , 6 , 1.0] 

        self.thruster_reallocation_sub = rospy.Subscriber(
            '/uuv0/thruster_reallocation', Float32MultiArray, self.callback_thruster_reallocation)
        self.thruster_reallocation = []

        # Subscribe to UUV HSD
        self.hsd_sub = rospy.Subscriber(
            '/uuv0/hsd_command', HSDCommand, self.callback_hsd_cmd)

        self.lec2_am_left_sub= rospy.Subscriber(
            "/uuv0/cm_am/left", Float32MultiArray, self.callback_lec2_am_left)
        self.lec2_am_right_sub= rospy.Subscriber(
            "/uuv0/cm_am/right", Float32MultiArray, self.callback_lec2_am_right)
        
        # self.lec_dd_am_sub= rospy.Subscriber(
        #     "/lec_dd_am/p_value", Float32MultiArray, self.callback_lec_dd_am)

        self.xtrack_error_sub = rospy.Subscriber(
            '/uuv0/xtrack_error', Float64, self.callback_xtrack_error) 

        self.waypoints_completed_sub = rospy.Subscriber(
            '/uuv0/waypoints_completed', Bool, self.callback_waypoints_completed)       

        self.counts = {
            "total" :               0,
            "pipeline_sls":         0,
            "pipeline_gt":          0,
            "pipeline_distance":    0,
            "wp_time":              0,
            "wp_complete":          -1
        }

        self.errors = {
            "pipeline_sls":         0,
            "semseg_not_used":      0,
            "pipeline_not_in_view": 0,
            "LEC2_not_triggered":   0,   
            "xtrack_total":         0        
        }

        value_arr = [collections.deque(maxlen=15) for _ in range(6)]
        self.values = {
                             'heading_change':              value_arr[0],
                             'lec2_right_logm':             value_arr[1],
                             'lec2_left_logm':              value_arr[2],
                             'lecdd_logm':                  value_arr[3],
                             'pipeline_distance':           value_arr[4],
                             'waypoints_complete':          value_arr[5]
                             }

        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            self.counts['total'] += 1
            last_used = (rospy.Time.now() - self.pipeline_in_view_msg.stamp).to_sec()
            last_seen = (rospy.Time.now() - self.pipeline_in_view_gt_msg.stamp).to_sec()
            
            # Check pipe last seen&used timestamp
            if last_used > 1.0:
                self.errors['semseg_not_used'] += 1  #"second"
            if last_seen > 1.0:
                self.errors['pipeline_not_in_view'] += 1  #"second"

            # Check if UUV heading change triggered LEC2 AM or not
            if (
                    len(self.values['heading_change']) == 15
                    and
                    len(self.values['lec2_right_logm']) > 0
                    and
                    len(self.values['lec2_left_logm']) > 0
            ): # deque initialized
                if (self.values['heading_change'][0] > 10): # heading change indicates obstacle avoidance
                    noise_threshold = 2.5
                    if (
                        np.max(self.values['lec2_right_logm']) < noise_threshold and
                        np.max(self.values['lec2_left_logm']) < noise_threshold
                    ):
                        self.errors['LEC2_not_triggered'] += 1 # [sec]
                        # print('Heading change DOES NOT triggered LEC2 AM')
                    else:
                        pass
                        # print('Heading change triggered LEC2 AM')
    
            if len(self.values['waypoints_complete']) == 15: # deque initialized       
                if (
                    self.counts['wp_complete'] < 0
                    and
                    self.values['waypoints_complete'][-1]
                    and
                    self.values['waypoints_complete'][-2]                
                ):
                    self.counts['wp_complete'] = self.counts['total'] 

            # if (
            #     self.counts['pipeline_gt'] > 0
            #     and 
            #     self.counts['pipeline_distance'] > 0
            #     and 
            #     self.counts['wp_time'] > 0
            # ):
            evaluation = self.eval()                
            # print(evaluation)
            self.write_eval(evaluation)
               
            rate.sleep() 
        rospy.on_shutdown(self.shutdown())


    def eval(self):
        str = ("""[bluerov_eval.py]

    Evaluation:
    =========
    * Simulation time total:    : %s [sec]
    ------ [ Pipe tracking metrics] ------
    * Pipeline detection ratio  : %s
    * Average pipeline distace  : %s [meter]
    * Tracking error ratio      : %s
    * Semseg bad, not used      : %s [sec]
    * Pipeline not in view      : %s [sec]
    * LEC2 AM not triggered     : %s [sec]
    -------- [ Waypoint metrics] --------
    * Average cross track error : %s [m]
    * Time to complete          : %s [sec]
    -------- [ Degradation GT info ] --------
    * Degradation starting time : %s [sec]
    * Degraded thruster id      : %s 
    * Degraded efficiency       : %s 
    -------- [ FDI LEC info ] --------
    * Thruster Reallocation     : %s [sec]
    * FDI Degraded thruster id  : %s 
    * FDI Degraded efficiency   : %s 

        """ % ( 
                self.counts['total'],
                self.counts['pipeline_sls'] / self.counts['pipeline_gt'] if self.counts['pipeline_gt'] > 0 else 0,
                self.values['pipeline_distance'][-1] / self.counts['pipeline_distance']  if self.counts['pipeline_distance'] > 0 else 0,
                self.errors['pipeline_sls'] / self.counts['pipeline_sls'] * 0.5  if self.counts['pipeline_sls'] > 0 else 0,
                self.errors['semseg_not_used'],
                self.errors['pipeline_not_in_view'],
                self.errors['LEC2_not_triggered'],
                self.errors['xtrack_total'] / self.counts['wp_time'] if self.counts['wp_time'] > 0 else 0,
                self.counts['wp_complete'],
                math.floor(self.degradation_gt[0]) if self.degradation_gt[0] != -1 else -1,
                self.degradation_gt[1] if self.degradation_gt[0] != -1 else -1,
                round(self.degradation_gt[2] * 100, 2) if self.degradation_gt[0] != -1 else -1,
                round(self.thruster_reallocation[0], 2) if len(self.thruster_reallocation) > 0 else -1,
                self.thruster_reallocation[1] if len(self.thruster_reallocation) > 0 else -1,
                round(self.thruster_reallocation[2] * 100, 2) if len(self.thruster_reallocation) > 0 else -1
                )
        )
        return str
    
    def shutdown(self):
        print("\n\n\n\n \033[1;32m WRITING AFTER SHUTDOWN \033[0m \n\n\n\n")
        self.write_eval(self.eval())

    def write_eval(self, eval_results):
        if os.path.isdir(results_dir):
            # rospy.loginfo("[Bluerov_Eval]\033[1;32m BlueROV evaluation writing results: " + os.path.join(results_dir, "bluerov_evaluation.txt \033[0m"))
            with open(os.path.join(results_dir, "bluerov_evaluation.txt"), 'w') as fd:
                fd.write(eval_results)                

    def callback_lec_dd_am(self, msg):
        pass
        # Todo:
        # self.values['lecdd_logm'].append(msg.values[0])
        
        # log(m):   lec_dd_am.values[0]
        # det:      lec_dd_am.values[2]

    def callback_lec2_am_left(self, msg):
        try:
            self.values['lec2_left_logm'].append(msg.values[0])
        except IndexError:
            pass
        
        # log(m):   lec2_am_left.values[0]
        # det:      lec2_am_left.values[1]

    def callback_lec2_am_right(self, msg):
        try:
            self.values['lec2_right_logm'].append(msg.values[0])
        except IndexError:
            pass
        
        # log(m):   lec2_am_right.values[0]
        # det:      lec2_am_right.values[1]

    def callback_hsd_cmd(self, msg):        
        self.values['heading_change'].append(msg.heading)

    def callback_obstacle_avoidance(self, msg):        
        pass

    def callback_pose(self, msg):
        pass
    
    def callback_pipeline_distance_sub(self, msg):
        # print('\033[1;32m callback_pipeline_distance: \033[0m \t' + str(abs(msg.data)) + "[m]")
        if len(self.values['pipeline_distance']) > 0:
            self.values['pipeline_distance'].append(self.values['pipeline_distance'].pop() + abs(msg.data))
        else:
            self.values['pipeline_distance'].append(abs(msg.data))
        self.counts['pipeline_distance'] += 1
        
    def callback_pipeline_in_view_sub(self, msg):
        self.pipeline_in_view_msg = msg
    def callback_pipeline_in_view_gt_sub(self, msg):
        self.pipeline_in_view_gt_msg = msg

    def callback_pipeline_in_sls_sub(self, msg):
        # print('\033[1;32m callback_pipeline_in_sls: \033[0m \t' + str(abs(0.5 - abs(msg.data))))        
        self.errors['pipeline_sls'] += abs(0.5 - abs(msg.data))
        self.counts['pipeline_sls'] += 1

    def callback_pipeline_in_gt_sub(self, msg):
        self.counts['pipeline_gt'] += 1
        # print('\033[1;32m callback_pipeline_in_gt: \033[0m \t' + str(abs(0.5 - abs(msg.data))))    

    def callback_xtrack_error(self, msg):
        if self.counts['wp_complete'] < 0:
            # Waypoints not finished
            self.errors['xtrack_total'] += abs(msg.data)
            self.counts['wp_time'] += 1
       
    def callback_waypoints_completed(self, msg):
        self.values['waypoints_complete'].append(msg.data)

    def callback_degradation_gt(self, msg):
        # No degradation info yet
        if self.degradation_gt[0] == -1:
            # Check for degradation info
            degradation_msg = np.array(msg.data)
            if degradation_msg[0] < 6 and degradation_msg[1] < 1.0:
                # Catch degradation
                self.degradation_gt = [
                    rospy.Time.now().to_sec(),   # degradation starting time
                    degradation_msg[0], # degraded thruster
                    degradation_msg[1]  # degraded efficiency
                ]

    def callback_thruster_reallocation(self, msg):
        msg = np.array(msg.data)
        self.thruster_reallocation.append(msg[0]) # Time
        self.thruster_reallocation.append(msg[1]) # Thruster ID from FDI LEC
        self.thruster_reallocation.append(msg[2]) # Efficiency from FDI LEC

if __name__=='__main__':
    rospy.init_node('bluerov_evaluation', log_level=rospy.INFO)
    results_dir = rospy.get_param("~results_directory")
    try:
        node = BluerovEval()
        rospy.spin()
    except rospy.ROSInterruptException:
        print('caught exception')
