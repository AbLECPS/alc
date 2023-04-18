#!/usr/bin/env python

import rospy
import numpy as np
import math
import os
import rospkg
import collections
import time

from rospy.numpy_msg import numpy_msg
from std_msgs.msg import Float32MultiArray
from vandy_bluerov.srv import TAM
from uuv_thruster_manager.srv import ThrusterManagerInfo, GetThrusterManagerConfig, SetThrusterManagerConfig
from uuv_gazebo_ros_plugins_msgs.msg import FloatStamped
from uuv_control_interfaces.dp_controller_local_planner import Hold
from uuv_control_msgs.srv import *


class ThrusterFunction(object):
    def __init__(self):
        self.namespace = rospy.get_namespace().replace('/', '')

        self.topic_suffix_default = "/input"
        self.topic_suffix_active_diagnostics = "/active_diagnostics_input"

        self.thruster_0_pub = rospy.Publisher(
            '/uuv0/thrusters/0' + self.topic_suffix_default, FloatStamped, queue_size=1)
        self.thruster_1_pub = rospy.Publisher(
            '/uuv0/thrusters/1' + self.topic_suffix_default, FloatStamped, queue_size=1)
        self.thruster_2_pub = rospy.Publisher(
            '/uuv0/thrusters/2' + self.topic_suffix_default, FloatStamped, queue_size=1)
        self.thruster_3_pub = rospy.Publisher(
            '/uuv0/thrusters/3' + self.topic_suffix_default, FloatStamped, queue_size=1)
        self.thruster_4_pub = rospy.Publisher(
            '/uuv0/thrusters/4' + self.topic_suffix_default, FloatStamped, queue_size=1)
        self.thruster_5_pub = rospy.Publisher(
            '/uuv0/thrusters/5' + self.topic_suffix_default, FloatStamped, queue_size=1)

        # rate = rospy.Rate(1)
        # while not rospy.is_shutdown():
        #         rate.sleep()

    def emergency_stop(self):
        # Thruster config is very slow...
        # store the original TAM
        _tam = self.get_TAM()
        _tam[:, 0:4] *= 0
        # set it only for vertical motion -> surface
        if not self.set_TAM(_tam):
            print("cannot set TAM")

        return True
        

        # _thruster_config = self.get_thruster_config()
        # response = self.set_thruster_topic_suffix(_thruster_config, self.topic_suffix_active_diagnostics)
        # if not response:
        #     return False
        # print("Manual thrust activated")    
        
        # # Turn back on thrusters with the original TAM
        # if not self.set_TAM(_tam):
        #     print("cannot set TAM")

        # # Reverse thrust for e-stop
        # test_input = 500.0
        # self.manual_reverse_thrust(test_input)
        
        # # 2 seconds reverse thrust to stop and clear obstacle
        # time.sleep(5.0)

        # # Command UUV to go straight up vertically
        # self.manual_surface_thrust(test_input)

        # # self.test_thrusters(0.0)
     
        # return True
 

    def restore_thruster_topics(self):
        _thruster_config = self.get_thruster_config()
        response = self.set_thruster_topic_suffix(_thruster_config, self.topic_suffix_default)
        if not response:
            return False
        print("Automatic thrust restored")    
        return True

    def active_diagnostics(self):
        return True
        # if self.active_diagnostics_enable:
        #     self.test_thrusters(0.0)

        #     # store the original TAM
        #     _tam = self.get_TAM()

        #     _thruster_config = self.get_thruster_config()
        #     response = self.set_thruster_topic_suffix(_thruster_config, self.topic_suffix_active_diagnostics)

        #     test_input = 500.0
        #     self.test_thrusters(test_input)

        #     detection_start_time = rospy.Time.now()
        #     detect_rate = rospy.Rate(1)
        #     detection_iteration = 0
        #     thruster_efficiency = np.zeros(6)
        #     while rospy.Time.now() - detection_start_time < rospy.Duration(secs=10):
        #         if rospy.Duration(secs=2) < rospy.Time.now() - detection_start_time:
        #             for t in range(6):
        #                 thruster_efficiency[t] += self.get_thruster_efficiency_est(self.ann_input[t],
        #                                                                            self.ann_input[t + 6])
        #             detection_iteration += 1
        #             if detection_iteration == 7:
        #                 detection_start_time = rospy.Time.now()
        #                 self.test_thrusters(-test_input)
        #         detect_rate.sleep()

        #     print("============================")
        #     print("Mesasured thruster efficiency: " + str(thruster_efficiency / detection_iteration / 0.72))
        #     self.test_thrusters(0.0)

        #     # Reallocate TAM based on the original and the detected degradation
        #     _thruster_config = self.get_thruster_config()

        #     #
        #     # Set the degradation reallocation here
        #     #

        #     response = self.set_thruster_topic_suffix(_thruster_config, self.topic_suffix_default)
        #     if not self.set_TAM(_tam):
        #         print("cannot set TAM")

    def get_thruster_config(self):
        try:
            rospy.wait_for_service('uuv0/thruster_manager/get_config', timeout=5)
        except rospy.ROSException:
            raise rospy.ROSException('thruster_manager/get_config Service not available!')

        try:
            GetConfig_srv = rospy.ServiceProxy(
                'uuv0/thruster_manager/get_config',
                GetThrusterManagerConfig)
        except rospy.ServiceException(e):
            raise rospy.ROSException('Service call failed, error=' + e)

        return GetConfig_srv()

    def set_thruster_topic_suffix(self, thruster_config, thruster_topic_suffix):
        """
        Sets thruster topic for active diagnostics
        /active_diagnostics_input
        /input
        """

        # Set the thruster topics
        try:
            rospy.wait_for_service('uuv0/thruster_manager/set_config', timeout=5)
        except rospy.ROSException:
            raise rospy.ROSException('thruster_manager/set_config Service not available!')
            return False

        try:
            SetConfig_srv = rospy.ServiceProxy(
                'uuv0/thruster_manager/set_config',
                SetThrusterManagerConfig)
        except rospy.ServiceException(e):
            raise rospy.ROSException('Service call failed, error=' + e)
            return False

        rospy.loginfo(SetConfig_srv(base_link=thruster_config.base_link,
                                    thruster_frame_base=thruster_config.thruster_frame_base,
                                    thruster_topic_prefix=thruster_config.thruster_topic_prefix,
                                    thruster_topic_suffix=thruster_topic_suffix,
                                    timeout=thruster_config.timeout))

        # reset DP Controller
        try:
            rospy.wait_for_service('uuv0/reset_controller', timeout=5)
        except rospy.ROSException:
            raise rospy.ROSException('reset_controller Service not available!')
            return False

        try:
            ResetController_srv = rospy.ServiceProxy(
                'uuv0/reset_controller',
                ResetController)
        except rospy.ServiceException(e):
            raise rospy.ROSException('Service call failed, error=' + e)
            return False

        print(ResetController_srv())
        
        return True

    def get_TAM(self):
        try:
            rospy.wait_for_service('uuv0/thruster_manager/get_thrusters_info', timeout=5)
        except rospy.ROSException:
            raise rospy.ROSException('thruster_manager/get_thrusters_info Service not available!')

        try:
            ThrusterManagerInfo_srv = rospy.ServiceProxy(
                'uuv0/thruster_manager/get_thrusters_info',
                ThrusterManagerInfo)
        except rospy.ServiceException(e):
            raise rospy.ROSException('Service call failed, error=' + e)

        response = ThrusterManagerInfo_srv()
        if len(response.allocation_matrix) == 36:
            return np.reshape(response.allocation_matrix, (6, 6))
        else:
            return False

    def set_TAM(self, tam):
        try:
            rospy.wait_for_service('uuv0/thruster_manager/set_tam', timeout=5)
        except rospy.ROSException:
            raise rospy.ROSException('thruster_manager/set_tam Service not available!')

        try:
            TAM_reallocation_srv = rospy.ServiceProxy(
                'uuv0/thruster_manager/set_tam',
                TAM)
        except rospy.ServiceException(e):
            raise rospy.ROSException('Service call failed, error=' + e)

        response = TAM_reallocation_srv(tam.flatten())
        print("TAM_reallocation_srv: " + str(response))

        return True

    def hold_vehicle(self):
        try:
            rospy.wait_for_service('uuv0/hold_vehicle', timeout=5)
        except rospy.ROSException:
            raise rospy.ROSException('Service not available!')

        try:
            hold_vehicle_srv = rospy.ServiceProxy(
                'uuv0/hold_vehicle',
                Hold)
        except rospy.ServiceException(e):
            raise rospy.ROSException('Service call failed, error=' + e)

        print(hold_vehicle_srv())

    def test_thrusters(self, input_thrust):
        msg = FloatStamped()
        msg.header.stamp = rospy.Time.now()
        msg.data = input_thrust
        self.thruster_0_pub.publish(msg)
        self.thruster_1_pub.publish(msg)
        self.thruster_2_pub.publish(msg)
        self.thruster_3_pub.publish(msg)
        self.thruster_4_pub.publish(msg)
        self.thruster_5_pub.publish(msg)

    def manual_reverse_thrust(self, input_thrust):
        msg = FloatStamped()
        msg.header.stamp = rospy.Time.now()
        msg.data = -input_thrust
        self.thruster_0_pub.publish(msg)
        self.thruster_1_pub.publish(msg)
        
        msg.data = input_thrust
        self.thruster_2_pub.publish(msg)
        self.thruster_3_pub.publish(msg)
        
        msg.data = 0
        self.thruster_4_pub.publish(msg)
        self.thruster_5_pub.publish(msg)

    def manual_surface_thrust(self, input_thrust):
        msg = FloatStamped()
        msg.header.stamp = rospy.Time.now()
        msg.data = 0
        self.thruster_0_pub.publish(msg)
        self.thruster_1_pub.publish(msg)
        self.thruster_2_pub.publish(msg)
        self.thruster_3_pub.publish(msg)
        
        msg.data = input_thrust
        self.thruster_4_pub.publish(msg)
        self.thruster_5_pub.publish(msg)

    def get_thruster_efficiency_est(self, x, y):
        if x > 0:
            p00 = 2.646
            p10 = -0.007444
            p01 = -0.002218
            p20 = 4.943e-06
            p11 = 6.275e-06
            p02 = 1.717e-07
            p21 = -3.689e-09
            p12 = -4.989e-10
            p03 = 5.947e-11
        else:
            p00 = 0.6765
            p10 = 0.002626
            p01 = 9.313e-05
            p20 = 2.787e-06
            p11 = 1.735e-08
            p02 = 1.703e-07
            p21 = -1.003e-09
            p12 = -1.948e-10
            p03 = -3.89e-11
        return p00 + p10 * x + p01 * y + p20 * x ** 2 + p11 * x * y + p02 * y ** 2 + p21 * x ** 2 * y + p12 * x * y ** 2 + p03 * y ** 3

    def sensor_fault_rpm_callback(self, msg):
        # From CM
        self.sensor_failure_rpm = msg.data
        return False

    def callback(self, msg):
        self.ann_input = msg.data[0:13]
        self.am_input = msg

    def active_diagnostics_callback(self, msg):
        self.active_diagnostics_enable = msg.data


if __name__ == '__main__':
    print('Starting Degradation Detector')
    # rospy.init_node('task_surface', log_level=rospy.DEBUG)
    rospy.init_node('degradation_detector', log_level=rospy.INFO)
    try:
        node = ThrusterFunction()
        rospy.spin()
    except rospy.ROSInterruptException:
        print('caught exception')
