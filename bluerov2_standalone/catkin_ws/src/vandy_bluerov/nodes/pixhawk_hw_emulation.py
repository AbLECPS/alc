#!/usr/bin/env python

# Description:  Emulation pixhawk (BlueROV2) sensor data for ALC simulation


import rospy
import numpy as np
import math
from std_srvs.srv import Empty
from vandy_bluerov.msg import HSDCommand
from vandy_bluerov.msg import PixhawkHW
from nav_msgs.msg import Odometry

from uuv_gazebo_ros_plugins_msgs.msg import FloatStamped
from std_msgs.msg import Float64
from message_filters import ApproximateTimeSynchronizer, Subscriber

class PixhawkHwEmulation(object):
    def __init__(self):
        self.namespace = rospy.get_namespace().replace('/', '')
        rospy.loginfo('Initialize Pixhawk (BlueROV2) sensor data for <%s>' % self.namespace)
        
        self.batt_voltage_full = rospy.get_param('~batt_voltage', 16.8) #4x3.7 nominal, 4x4.2 fully charged
        self.batt_capacity = rospy.get_param('~batt_capacity', 18.0)
        self.batt_charge = rospy.get_param('~batt_charge', 1.0) 
        self.failed_rpm_sensor = rospy.get_param('~failed_rpm_sensor', 2) 
        self.failed_rpm_sensor_enable = rospy.get_param('~failed_rpm_sensor_enable', False)
        self.failed_rpm_sensor_start = rospy.get_param('~failed_rpm_sensor_start', 50)
        #lipo batt:
        self.cell_max_voltage = 4.2
        self.cell_min_voltage = 3.0
        self.batt_voltage_remaining = (self.batt_voltage_full / self.cell_max_voltage) * \
            (self.batt_charge  * (self.cell_max_voltage - self.cell_min_voltage) + self.cell_min_voltage)
        self.batt_capacity_remaining = self.batt_capacity * self.batt_charge
        self.batt_charge_remaining = self.batt_charge
        self.thruster_failure = rospy.get_param('~thruster_failure', -1)
        rospy.loginfo('Battery remaining capacity:  %0.2f Ah | Battery remaining voltage:  %0.2f V | Battery charge:  %0.2f' 
                %(self.batt_capacity_remaining, self.batt_voltage_remaining, self.batt_charge))
                
        # # Reading the thruster input topic prefix
        # self.thruster_topic_prefix = rospy.get_param('~thruster_topic_prefix', 'thruster')
        # self.thruster_topic_suffix = rospy.get_param('~thruster_topic_suffix', 'input')
        
        # self.sub_thruster = list()

        # for i in range(6):
        #     topic = '%s/%d/%s' % (self.thruster_topic_prefix, i, self.thruster_topic_suffix)
        #     self.sub_thruster.append(Subscriber(topic, FloatStamped)

        self.thruster_0_thrust_sub = Subscriber('thrusters/0/thrust', FloatStamped)
        self.thruster_1_thrust_sub = Subscriber('thrusters/1/thrust', FloatStamped)
        self.thruster_2_thrust_sub = Subscriber('thrusters/2/thrust', FloatStamped)
        self.thruster_3_thrust_sub = Subscriber('thrusters/3/thrust', FloatStamped)
        self.thruster_4_thrust_sub = Subscriber('thrusters/4/thrust', FloatStamped)
        self.thruster_5_thrust_sub = Subscriber('thrusters/5/thrust', FloatStamped)

        approxTimeSync=ApproximateTimeSynchronizer([self.thruster_0_thrust_sub,
                                                    self.thruster_1_thrust_sub,
                                                    self.thruster_2_thrust_sub,
                                                    self.thruster_3_thrust_sub,
                                                    self.thruster_4_thrust_sub,
                                                    self.thruster_5_thrust_sub], queue_size=1, slop=0.1)

        approxTimeSync.registerCallback(self.calc_amps)

        self.pub = rospy.Publisher('pixhawk_hw', PixhawkHW, queue_size=1)                                                   

    # def calc_amps(self, msg):
    def calc_amps(self,*args):
        '''
            Receives thrust message, then:
                - calculates power consumption
                - simulates RPM sensor
                - simulates RPM sensor failure
        '''
        # 20 Hz
        self.rpm=[]
        self.sum_thrust=0.0
        i=0
        for thrust in args:            
            # Polinomial fit for T200 thruster power draw [A] based on public performance data with 4S LiIon (14.8V) battery            
            if not (
                i == self.failed_rpm_sensor 
                and self.failed_rpm_sensor_enable 
                and (rospy.Time.now() > rospy.Time(self.failed_rpm_sensor_start))
            ):
                self.sum_thrust += thrust.data**2 * 0.6084 + abs(thrust.data) * 1.7861 - 0.3008
            # else:
            #     print('[--==[ RPM SENSOR FAIL ]==--]')
            self.rpm.append(self.calc_rpm(thrust.data))
            # rospy.loginfo(thrust.data)
            i += 1
        thrusters_power = max(self.sum_thrust,0) 
        self.batt_capacity_remaining -= thrusters_power * 0.05 / 3600
        self.batt_voltage_remaining = (self.batt_capacity_remaining / self.batt_capacity) * self.batt_voltage_full
        self.batt_charge_remaining = (self.batt_capacity_remaining / self.batt_capacity)
        # rospy.loginfo('Thrusters power: %0.2f A | Remaining Ah:  %0.2f Ah | Remaining voltage:  %0.2f V | Battery charge:  %0.2f' 
        #         %(thrusters_power, self.batt_capacity_remaining, self.batt_voltage_remaining, self.batt_charge_remaining))
        msg = PixhawkHW(header = args[0].header,
                        thrusters_power=thrusters_power,
                        batt_capacity_remaining=self.batt_capacity_remaining,
                        batt_voltage_remaining=self.batt_voltage_remaining,
                        batt_charge_remaining=self.batt_charge_remaining,
                        rpm=self.rpm)
        self.pub.publish(msg)
        # rospy.loginfo(msg.rpm)

    def calc_rpm(self, thrust):
        if thrust == 0:
            return 0
        elif thrust < 0:
            return abs(round(55.76 * thrust**3 + 476.8 * thrust**2 + 1775 * thrust - 436.9, 2))
        else:
            return abs(round(28.57 * thrust**3 - 307.6 * thrust**2 + 1435 * thrust + 406.3, 2))
        

if __name__=='__main__':
    print('Starting Pixhawk (BlueROV2) sensor data emulation for ALC simulation')
    rospy.init_node('pixhawk_hw_emulation')
    try:
        node = PixhawkHwEmulation()
        rospy.spin()
    except rospy.ROSInterruptException:
        print('caught exception')
