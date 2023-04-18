#!/usr/bin/env python
import rospy
import os
import time
from dronekit import connect, VehicleMode
import time
from pymavlink import mavutil
from datetime import datetime
from message_filters import ApproximateTimeSynchronizer, Subscriber
from uuv_gazebo_ros_plugins_msgs.msg import FloatStamped
import json

class MavlinkInterface(object):
    """
    """
    def __init__(self):
        # connection_string = 'udp:10.42.0.1:14550'
        connection_string = 'udp:0.0.0.0:14550'
        print('Connecting to vehicle on: %s' % connection_string)

        self.vehicle = connect(connection_string, wait_ready=True)

        self.vehicle.wait_ready('autopilot_version')

        print(" ---===[ ArduPilot Connected ]===---")
        print(" Autopilot Firmware version: %s" % self.vehicle.version)
        print(" Mode: %s" % self.vehicle.mode.name ) 

        self.set_servo_center()

        self.set_servo_function()
        print("Arming motors")
        self.vehicle.armed = True

        while not self.vehicle.armed:
            print(" Waiting for arming...")
            self.vehicle.armed = True
            time.sleep(1)

        # Print the armed state for the vehicle
        print (" Armed: %s" % self.vehicle.armed)

        # Configure ATTITUDE message to be sent at 50Hz
        self.request_message_interval(mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE, 50)

        self.thruster_0_sub = Subscriber('uuv0/thrusters/0/input', FloatStamped)
        self.thruster_1_sub = Subscriber('uuv0/thrusters/1/input', FloatStamped)
        self.thruster_2_sub = Subscriber('uuv0/thrusters/2/input', FloatStamped)
        self.thruster_3_sub = Subscriber('uuv0/thrusters/3/input', FloatStamped)
        self.thruster_4_sub = Subscriber('uuv0/thrusters/4/input', FloatStamped)
        self.thruster_5_sub = Subscriber('uuv0/thrusters/5/input', FloatStamped)
        approxTimeSync=ApproximateTimeSynchronizer([self.thruster_0_sub,
                                                    self.thruster_1_sub,
                                                    self.thruster_2_sub,
                                                    self.thruster_3_sub,
                                                    self.thruster_4_sub,
                                                    self.thruster_5_sub
                                                    ], queue_size=1, slop=0.02)

        approxTimeSync.registerCallback(self.thruster_callback)
        self.thruster_callback_msg = []

        @self.vehicle.on_message('ATTITUDE')
        def listener(self, name, msg):
            print(msg.roll)
            # msg.roll
            # msg.pitch
            # msg.yaw
            # msg.rollspeed
            # msg.pitchspeed
            # msg.yawspeed    
            pass

        # rate = rospy.Rate(hz)
        # while True:
        #     pass

            # pwm = 1500 + 500*self.vehicle.attitude.roll
            # self.set_servo_output(1, pwm)
            # print(str(datetime.now().time())+": "+str(self.vehicle.attitude.roll))
            # time.sleep(0.02)

    def thruster_callback(self,*args):
        for idx, thruser_input in enumerate(args):                  
            pwm = 1500 + min(max(thruser_input.data/2, -400), 400)
            self.set_servo_output(idx+1, pwm)


    def set_servo_function(self):
        # Set servo function to 0 (disable) to control over mavlink
        self.vehicle.parameters['SERVO1_FUNCTION']=0
        self.vehicle.parameters['SERVO2_FUNCTION']=0
        self.vehicle.parameters['SERVO3_FUNCTION']=0
        self.vehicle.parameters['SERVO4_FUNCTION']=0
        self.vehicle.parameters['SERVO5_FUNCTION']=0
        self.vehicle.parameters['SERVO6_FUNCTION']=0
        self.vehicle.parameters['SERVO7_FUNCTION']=0
        self.vehicle.parameters['SERVO8_FUNCTION']=0

    def set_servo_output(self, srv_channel, pwm):
        msg = self.vehicle.message_factory.command_long_encode(
            0,0, #target_system, target_component
            mavutil.mavlink.MAV_CMD_DO_SET_SERVO, #Command
            0, #confirmation
            srv_channel, #param1, servo number
            pwm, #param2, PWM in microseconds - typically 1000 to 2000)
            0,0,0,0,0 # not used
            )
        # send command to vehicle
        self.vehicle.send_mavlink(msg)
        # print("MAV_CMD_DO_SET_SERVO")

    def request_message_interval(self, message_id, frequency_hz):
        """
        Request MAVLink message in a desired frequency,
        documentation for SET_MESSAGE_INTERVAL:
            https://mavlink.io/en/messages/common.html#MAV_CMD_SET_MESSAGE_INTERVAL

        Args:
            message_id (int): MAVLink message ID
            frequency_hz (float): Desired frequency in Hz
        """
        self.vehicle.send_mavlink(
            self.vehicle.message_factory.command_long_encode(
                0,0,
                mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
                message_id, # The MAVLink message ID
                1e6 / frequency_hz, # The interval between two messages in microseconds. Set to -1 to disable and 0 to request default rate.
                0, 0, 0, 0, # Unused parameters
                0, # Target address of message stream (if message has target address fields). 0: Flight-stack default (recommended), 1: address of requestor, 2: broadcast.
            )
        )
        print("MAV_CMD_SET_MESSAGE_INTERVAL")

    def set_servo_center(self):
        # PWM 1500us is servo center where ESCs are booting up
        for i in range(1,7):
            self.set_servo_output(i, 1500)

if __name__ == '__main__':
    rospy.init_node('mavlink_interface', log_level=rospy.INFO)
    try:
        node = MavlinkInterface()
        rospy.spin()
    except rospy.ROSInterruptException:
        print('caught exception')
