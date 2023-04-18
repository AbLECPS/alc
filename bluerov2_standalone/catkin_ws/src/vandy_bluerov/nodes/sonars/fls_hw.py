#!/usr/bin/env python
import roslib
import rospy
from std_msgs.msg import Float32MultiArray
import serial
import numpy as np

def serial_device():
    serial_port = rospy.get_param('~serial_port', '/dev/ttyACM0')
    serial_baud = int(rospy.get_param('~serial_baud', 921600))
    topic = rospy.get_param('~topic', '/vu_fls/bins')

    ser = serial.Serial(serial_port, serial_baud)


    pub = rospy.Publisher(
        topic, Float32MultiArray, queue_size=1)
    rospy.init_node('fls_serical_device')
    msg = Float32MultiArray()
    while not rospy.is_shutdown():
        # try:
            data = ser.readline().decode("utf-8")[1:-3] 
            data = np.array(data.split(", "), dtype='i')
            data -= np.min(data)
            data = np.multiply(data, 2.8).astype(int) # scaling to 0-255
            msg.data = data.flatten()
            pub.publish(msg)
            # print(msg)
        # except:
        #     print("Serial error")


if __name__ == '__main__':
    try:
        serial_device()
    except rospy.ROSInterruptException:
        pass