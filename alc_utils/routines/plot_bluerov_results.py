# Script which plots various commonly used metrics for UUV Sim data
# Uses topic names from Bluerov setup as default options

import rospy
import rosbag
import matplotlib.pyplot as plt
import numpy as np
import tf
import math
import matplotlib
import os
from matplotlib.lines import Line2D
import tf.transformations as trans

VEHICLE_ODOM_TOPIC =        "/uuv0/pose_gt_noisy_ned"
OBSTACLE_HSD_TOPIC =        "/uuv0/hsd_obstacle_avoidance"
FLS_RANGE_TOPIC =           "/uuv0/fls_echosunder"
LEC3_RANGE_TOPIC =          "/lec3lite/ranges"
LEC3_AM_TOPIC =             "/lec3lite/am_vae"
PIPE_DISTANCE_TOPIC =       "/uuv0/pipeline_distance_from_mapping"
LEC2_AM_L_TOPIC =           "/vu_sss/am_vae_lec2lite_l"
LEC2_AM_R_TOPIC =           "/vu_sss/am_vae_lec2lite_r"
# UUV_SPEED_TOPIC                    = "/uuv0/speed"
UUV_SPEED_CMD_TOPIC =       "/uuv0/hsd_command"
BATT_LEVEL_TOPIC =          "/uuv0/pixhawk_hw"
CM_FAILSAFE_TOPIC =         "/uuv0/cm_failsafe"
CM_STATE_TOPIC =            "/uuv0/cm_state_machine"
DD_LEC =                    "/uuv0/degradation_detector"
DEGRADATION_GT =            "/uuv0/degradation_gt"
DD_LEC_AM =                 "/lec_dd_am/p_value"
THRUSTERS_TOPIC =           "/uuv0/thruster_cmd_logging"
RESONATE_TOPIC =            "/hazard_rate"
REALLOCATION_TOPIC =        "/uuv0/thruster_reallocation"

WPT_MARKER_TOPIC =          "/uuv0/waypoint_markers"
OBSTACLE_MARKER_TOPIC =     "/spawn_box_obstacles/collision_objects"
PIPE_MARKER_TOPIC =         "/pipeline/plotmarker"


DEFAULT_TOPICS = {"odom":           VEHICLE_ODOM_TOPIC,
                  "obstacle_hsd":	OBSTACLE_HSD_TOPIC,
                  "fls_range":	    FLS_RANGE_TOPIC,  # RAW
                  "lec3_range":	    LEC3_RANGE_TOPIC,  
                  "lec3_am":        LEC3_AM_TOPIC,
                  "pipe_distance":	PIPE_DISTANCE_TOPIC,
                  "lec2_am_l":	    LEC2_AM_L_TOPIC,
                  "lec2_am_r":	    LEC2_AM_R_TOPIC,
                  # "uuv_speed":	UUV_SPEED_TOPIC,
                  "uuv_speed_cmd":	UUV_SPEED_CMD_TOPIC,
                  "batt_level":	    BATT_LEVEL_TOPIC,
                  "cm_failsafe":	CM_FAILSAFE_TOPIC,
                  "cm_state":	    CM_STATE_TOPIC,
                  "dd_lec":         DD_LEC,
                  "degradation_gt":	DEGRADATION_GT,
                  "dd_lec_am":	    DD_LEC_AM,
                  "thrusters":      THRUSTERS_TOPIC,
                  "resonate":       RESONATE_TOPIC,
                  "reallocation":   REALLOCATION_TOPIC,
                  "waypoints":      WPT_MARKER_TOPIC,
                  "obstacles":      OBSTACLE_MARKER_TOPIC,
                  "pipeline":       PIPE_MARKER_TOPIC
                }


def plot_results(bagfilename,
                 topic_dict=DEFAULT_TOPICS,
                 start_time=0):
    # Init bag file
    print("Loading bag file %s..." % bagfilename)
    ros_start_time = rospy.Time(start_time)
    bagfile = rosbag.Bag(bagfilename)
    print("Done loading bag file.")

    i = 0
    uuv_pose = []
    obstacles = []
    depth_error_times = []
    uuv_speed_data = []
    uuv_speed_cmd_data = []
    uuv_speed_times = []
    uuv_speed_cmd_times = []
    times = []
    sim_time = 300

    for topic_type, topic_name in topic_dict.iteritems():
        # Load topic messages from bag file
        # print "Loading topic (%s) starting from time t = %d seconds..." % (topic_name, start_time)
        msgs = bagfile.read_messages(
            topics=[topic_name], start_time=ros_start_time)
        # print "Done loading topic."

        # FIXME: Bag file reader returns a generator. Need another way to check if any messages are present, if desired.
        if msgs is None:
            print("No data on topic %s present in bag file." % topic_name)
            continue
        
        matplotlib.rcParams['figure.figsize'] = [15, 4]
        # Get relevant data fields from messages and plot data based on topic
        if topic_type == "odom":
            matplotlib.rcParams['figure.figsize'] = [15, 15]
            x_data = []
            y_data = []
            obstacles = np.array(obstacles)
            o = []
            depth = []
            for topic, msg, timestamp in msgs:
                timestamp_ms = (timestamp.secs * 1000.0) + \
                    (timestamp.nsecs / 1000000.0)
                if len(obstacles):
                    o_idx = np.where(obstacles[:,0]==timestamp.secs)[0]
                    if len(o_idx):
                        rpy = trans.euler_from_quaternion([msg.pose.pose.orientation.x,
                                        msg.pose.pose.orientation.y,
                                        msg.pose.pose.orientation.z,
                                        msg.pose.pose.orientation.w])     
                        o.append([
                            msg.pose.pose.position.x + obstacles[o_idx,1] * np.cos(rpy[2]),
                            msg.pose.pose.position.y + obstacles[o_idx,1] * np.sin(rpy[2])
                        ])
                        obstacles = np.delete(obstacles, o_idx, 0)
                depth_error_times.append(timestamp_ms / 1000)
                x_data.append(msg.pose.pose.position.x)
                y_data.append(msg.pose.pose.position.y)
                depth.append(-msg.pose.pose.position.z)
                uuv_speed_data.append(vector_to_mag(msg.twist.twist.linear))
                sim_time = timestamp.secs

            if (len(x_data) and len(y_data)):
                plt.figure(17)
                plt.plot(y_data, x_data, color="orange", label='UUV path')
                plt.grid()
                plt.axis('equal')
                plt.xlabel("Position (m)")
                plt.ylabel("Position (m)")
                plt.title("Top view position", fontsize=14, fontweight='bold')
            
            if len(o):                
                plt.figure(17)
                plt.plot(np.array(o)[:,1], np.array(o)[:,0], 'rs',
                         markersize=20, label='Obstacles')
                plt.legend()    

            matplotlib.rcParams['figure.figsize'] = [15, 4]
            if (len(depth_error_times) and len(depth)):
                plt.figure(18)
                plt.plot(depth_error_times, depth)
                plt.xlabel("Time (s)")
                plt.grid()
                plt.ylabel("Depth (m)")
                plt.ylim(-50, 0)
                plt.title("UUV Depth", fontsize=14, fontweight='bold')

        if topic_type == "obstacles":
            matplotlib.rcParams['figure.figsize'] = [15, 15]
            times = []
            for topic, msg, ts in msgs:
                obstacles.append([ts.secs, msg.pose.position.x, msg.pose.position.y])  
                  
                
            # if len(data):                
            #     plt.figure(17)
            #     plt.plot(np.array(data)[:,0], np.array(data)[:,1], 's',
            #              markersize=20, label='Obstacles')
            #     plt.legend()         

        
        if topic_type == "waypoints":
            matplotlib.rcParams['figure.figsize'] = [15, 15]
            data = []            
            times = []
            initial = True
            for topic, msg, _ in msgs:
                data = msg.markers  
                if initial:
                    initial = False
                    if len(data):                
                        plt.figure(17)
                        x = []
                        y = []
                        for i in range(len(data)):
                            x.append(data[i].pose.position.x)  
                            y.append(data[i].pose.position.y)                  
                        plt.plot(x, y, 'kP',
                                markersize=20, label='Initial waypoints')
                        plt.legend()           
        
            if len(data):                
                plt.figure(17)
                x = []
                y = []
                for i in range(len(data)):
                    x.append(data[i].pose.position.x)  
                    y.append(data[i].pose.position.y)                  
                plt.plot(x, y, 'rX',
                         markersize=14, label='Final waypoints')
                plt.legend()           
        

        if topic_type == "pipeline":
            matplotlib.rcParams['figure.figsize'] = [15, 15]
            data = []
            times = []
            for topic, msg, _ in msgs:
                data.append(msg.data)                
                
            if len(data):
                data=np.array(data)
                data=data.reshape((np.size(data)/2, 2))
                plt.figure(17)                               
                for i in range(len(data)-1):                    
                    plt.plot([data[i,0], data[i+1,0]], [data[i,1], data[i+1,1]], 'c')
                # plt.legend()              
        
        elif topic_type == "obstacle_hsd":
            data = []
            times = []
            for topic, msg, timestamp in msgs:
                timestamp_ms = (timestamp.secs * 1000.0) + \
                    (timestamp.nsecs / 1000000.0)
                times.append(timestamp_ms / 1000)
                data.append(msg.heading)

            if (not(len(times))):
                continue

            plt.figure(1)
            plt.plot(times, data)
            plt.xlabel("Time (s)")
            plt.grid()
            plt.ylabel("Heading CMD")
            plt.title("Obstacle avoidance Heading CMD (0: no obstacle)",
                      fontsize=14, fontweight='bold')

        elif (topic_type == "fls_range"):
            data = []
            times = []
            for topic, msg, timestamp in msgs:
                timestamp_ms = (timestamp.secs * 1000.0) + \
                    (timestamp.nsecs / 1000000.0)
                times.append(timestamp_ms / 1000)
                data.append(msg.range)

            if (not(len(times))):
                continue

            plt.figure(0)
            plt.plot(times, data, label='BlueROV fls_echosunder')
            plt.xlabel("Time (s)")
            plt.grid()
            plt.ylabel("FLS range (m)")
            plt.title("FLS obstacle range", fontsize=14, fontweight='bold')

        elif (topic_type == "lec3_range"):
            data = []
            times = []
            for topic, msg, timestamp in msgs:
                timestamp_ms = (timestamp.secs * 1000.0) + \
                    (timestamp.nsecs / 1000000.0)
                times.append(timestamp_ms / 1000)
                if 0 < np.min(msg.data) < 30:
                    data.append(np.min(msg.data))
                else:
                    data.append(-1)
            if (not(len(times))):
                continue

            plt.figure(0)
            plt.plot(times, data, 'o', label='LEC3Lite FLS')
            _, ymax = plt.ylim()
            plt.ylim(-0.1, ymax)
            plt.legend()
            plt.xlabel("Time (s)")
            plt.ylabel("FLS range (m)")
            plt.title("LEC3Lite obstacle range", fontsize=14, fontweight='bold')

        elif topic_type == "pipe_distance":
            data = []
            times = []
            for topic, msg, timestamp in msgs:
                timestamp_ms = (timestamp.secs * 1000.0) + \
                    (timestamp.nsecs / 1000000.0)
                times.append(timestamp_ms / 1000)
                data.append(msg.data)

            if (not(len(times))):
                continue

            plt.figure(5)
            plt.plot(times, data)
            plt.xlabel("Time (s)")
            plt.grid()
            plt.ylabel("Pipe distance (m)")
            plt.title("UUV - Pipe distance", fontsize=14, fontweight='bold')

        elif topic_type == "lec3_am":            
            log_m = []
            det = []
            times = []
            for topic, msg, timestamp in msgs:
                if (np.array(msg.data).size) > 0:
                    timestamp_ms = (timestamp.secs * 1000.0) + \
                        (timestamp.nsecs / 1000000.0)
                    times.append(timestamp_ms / 1000)
                    log_m.append(msg.data[-2])
                    det.append(msg.data[-1])

            if (not(len(times))):
                continue

            plt.figure(2)
            ax = plt.subplot(2, 1, 1)
            plt.title("LEC3 Assurance monitor output - log(martingale))",
                      fontsize=14, fontweight='bold')
            plt.plot(times, log_m, label='log(martingale)')
            plt.ylim(-3, 25)
            plt.grid()
            # plt.legend()
            plt.subplot(2, 1, 2, sharex=ax)
            plt.title("LEC3 Assurance monitor output - detector",
                      fontsize=14, fontweight='bold')
            plt.plot(times, det, label='detector')
            _, xmax = plt.xlim()
            plt.xlim(0, xmax)
            # plt.legend()
            plt.xlabel("Time (s)")
            plt.grid()
            plt.ylabel(" ")


        elif topic_type == "lec2_am_l":            
            log_m = []
            det = []
            times = []
            for topic, msg, timestamp in msgs:
                if (np.array(msg.data).size) > 0:
                    timestamp_ms = (timestamp.secs * 1000.0) + \
                        (timestamp.nsecs / 1000000.0)
                    times.append(timestamp_ms / 1000)
                    log_m.append(msg.data[-2])
                    det.append(msg.data[-1])

            if (not(len(times))):
                continue

            plt.figure(3)
            ax = plt.subplot(2, 1, 1)
            plt.title("LEC2 Left Assurance monitor output - log(martingale))",
                      fontsize=14, fontweight='bold')
            plt.plot(times, log_m, label='log(martingale)')
            plt.ylim(-3, 25)
            plt.grid()
            # plt.legend()
            plt.subplot(2, 1, 2, sharex=ax)
            plt.title("LEC2 Left Assurance monitor output - detector",
                      fontsize=14, fontweight='bold')
            plt.plot(times, det, label='detector')
            _, xmax = plt.xlim()
            plt.xlim(0, xmax)
            # plt.legend()
            plt.xlabel("Time (s)")
            plt.grid()
            plt.ylabel(" ")

        elif topic_type == "lec2_am_r":
            log_m = []
            det = []
            times = []
            for topic, msg, timestamp in msgs:
                if (np.array(msg.data).size) > 0:
                    timestamp_ms = (timestamp.secs * 1000.0) + \
                        (timestamp.nsecs / 1000000.0)
                    times.append(timestamp_ms / 1000)
                    log_m.append(msg.data[-2])
                    det.append(msg.data[-1])

            if (not(len(times))):   
                continue

            plt.figure(4)
            ax = plt.subplot(2, 1, 1)
            plt.title("LEC2 Right Assurance monitor output - log(martingale)",
                      fontsize=14, fontweight='bold')
            plt.plot(times, log_m, label='log(martingale)')
            plt.ylim(-3, 25)
            plt.grid()
            # plt.legend()
            plt.subplot(2, 1, 2, sharex=ax)
            plt.title("LEC2 Right Assurance monitor output - detector",
                      fontsize=14, fontweight='bold')
            plt.plot(times, det, label='detector')
            _, xmax = plt.xlim()
            plt.xlim(0, xmax)
            # plt.legend()
            plt.xlabel("Time (s)")
            plt.grid()
            plt.ylabel(" ")

        # elif topic_type == "uuv_speed_cmd":
        #     for topic, msg, timestamp in msgs:
        #         timestamp_ms = (timestamp.secs * 1000.0) + \
        #             (timestamp.nsecs / 1000000.0)
        #         uuv_speed_cmd_times.append(timestamp_ms / 1000)
        #         uuv_speed_cmd_data.append(msg.speed)

        #     if (not(len(times))):
        #         continue

        #     plt.figure(20)
        #     plt.plot(depth_error_times, uuv_speed_data, label='Speed')
        #     plt.plot(uuv_speed_cmd_times,
        #              uuv_speed_cmd_data, label='Speed CMD')
        #     plt.legend()
        #     plt.grid()
        #     plt.xlabel("Time (ms)")
        #     plt.ylabel("Speed CMD; Speed (m/s)")
        #     plt.title("Speed and Speed command based on LEC2AM",
        #               fontsize=14, fontweight='bold')

        elif topic_type == "batt_level":
            data = []
            times = []
            for topic, msg, timestamp in msgs:
                timestamp_ms = (timestamp.secs * 1000.0) + \
                    (timestamp.nsecs / 1000000.0)
                times.append(timestamp_ms / 1000)
                data.append(msg.batt_charge_remaining)

            if (not(len(times))):
                continue

            plt.figure(16)
            plt.plot(times, data)
            plt.xlabel("Time (s)")
            plt.grid()
            plt.ylabel("Battery level")
            plt.title("Battery level", fontsize=14, fontweight='bold')

        elif topic_type == "resonate":
            data = []
            times = []
            for topic, msg, timestamp in msgs:
                timestamp_ms = (timestamp.secs * 1000.0) + \
                    (timestamp.nsecs / 1000000.0)
                times.append(timestamp_ms / 1000)
                data.append(msg.data)

            if (not(len(times))):
                continue

            plt.figure(30)
            plt.plot(times, data)
            plt.xlabel("Time (s)")
            plt.grid()
            plt.ylabel("Risk level")
            plt.title("ReSoNate risk level", fontsize=14, fontweight='bold')
            _, xmax = plt.xlim()
            plt.xlim(0, xmax)

        elif topic_type == "degradation_gt":
            degraded_thruster = []
            efficiency = []
            times = []
            for topic, msg, timestamp in msgs:
                timestamp_ms = (timestamp.secs * 1000.0) + \
                    (timestamp.nsecs / 1000000.0)
                times.append(timestamp_ms / 1000)
                degraded_thruster.append(msg.data[0])
                efficiency.append(msg.data[1])

            if (not(len(times))):
                continue

            plt.figure(7)
            plt.plot(times, degraded_thruster, label='Ground truth')
            plt.legend()
            # plt.xlabel("Time (s)")
            plt.grid()
            # plt.ylabel("Ground truth degraded thruster (0 to 5, 6 is nominal)")
            # plt.title("Ground truth degradation 1: Thruster ID", fontsize=14, fontweight='bold')

            plt.figure(8)
            plt.plot(times, efficiency, label='Ground truth')
            plt.legend()
            # plt.xlabel("Time (s)")
            plt.grid()
            # plt.ylabel("Ground truth efficiency (0.0 to 1.0)")

        elif topic_type == "dd_lec":
            est_class = []
            est_thr_id = []
            est_thr_eff = []
            est_decision_s = []
            est_decision_c = []
            est_decision_sx = []
            times = []
            for topic, msg, timestamp in msgs:
                timestamp_ms = (timestamp.secs * 1000.0) + \
                    (timestamp.nsecs / 1000000.0)
                times.append(timestamp_ms / 1000)
                est_class.append(msg.data[2])
                est_thr_id.append(msg.data[0])
                est_thr_eff.append(msg.data[1])
                est_decision_s.append(msg.data[5])
                est_decision_c.append(msg.data[6])
                est_decision_sx.append(msg.data[7])

            if (not(len(times))):
                continue

            plt.figure(6)
            plt.plot(times, est_class, 'o', label='LEC estimate')
            # plt.legend()
            plt.title("Degradation Detector (FDIR) LEC Class estimate",
                      fontsize=14, fontweight='bold')
            plt.ylabel(" ")
            plt.xlabel("Time (s)")
            plt.grid()
            plt.ylim(-1, 22)

            plt.figure(7)
            plt.plot(times, est_thr_id, 'o', label='LEC estimate')
            plt.legend()
            plt.title("Degradation Detector (FDIR) LEC Faulty Thruster ID estimate",
                      fontsize=14, fontweight='bold')
            plt.ylabel(" ")
            plt.xlabel("Time (s)")

            plt.figure(8)
            plt.plot(times, est_thr_eff, 'o', label='LEC estimate')
            plt.legend()
            plt.title("Degradation Detector (FDIR) LEC Faulty Thruster efficiency estimate",
                      fontsize=14, fontweight='bold')
            plt.ylabel("Efficiency (0.0 to 1.0)")
            plt.xlabel("Time (s)")
            plt.ylim(-0.1, 1.1)

            plt.figure(9)
            plt.plot(times, est_decision_s, label='Snapshot AM Decision')
            plt.plot(times, est_decision_c, label='Combined AM Decision')
            plt.plot(times, est_decision_sx, label='LEC Softmax Decision')
            plt.title("Degradation Detector (FDIR) LEC & AM decision outputs",
                      fontsize=14, fontweight='bold')
            plt.ylabel(" ")
            plt.xlabel("Time (s)")
            plt.grid()
            plt.legend()

        elif topic_type == "reallocation":

            data = []
            times = []
            for topic, msg, timestamp in msgs:
                timestamp_ms = (timestamp.secs * 1000.0) + \
                    (timestamp.nsecs / 1000000.0)
                times.append(timestamp_ms / 1000)
                data.append(msg.data[2])

            if (not len(times)):
                continue
            plt.figure(8)
            plt.plot(times, data, 'rD', markersize=15,
                     label='Reallocation marker')
            plt.legend()

            plt.figure(7)
            plt.plot(times, 6.1, 'rv', markersize=15,
                     label='Reallocation marker')
            plt.plot(times, -0.1, 'r^', markersize=15,)
            plt.legend()

            plt.figure(9)
            plt.plot(times, 1.1, 'rv', markersize=15,
                     label='Reallocation marker')
            plt.plot(times, -0.1, 'r^', markersize=15,)
            plt.legend()

        elif topic_type == "thrusters":
            times = []
            data = []
            for topic, msg, timestamp in msgs:
                timestamp_ms = (timestamp.secs * 1000.0) + \
                    (timestamp.nsecs / 1000000.0)
                times.append(timestamp_ms / 1000)
                data.append(msg.data[0:12])

            if (not(len(times))):
                continue

            plt.figure(50)
            thrusters_data = list(zip(*data))
            for i in range(6):
                plt.subplot(6, 1, i+1)
                plt.title("Thruster #" + str(i),
                          fontsize=14, fontweight='bold')
                plt.plot(times, thrusters_data[i], label='CMD')
                plt.plot(times, thrusters_data[i + 6], label='RPM')
                plt.ylim(-5000, 5000)
                plt.legend()

            plt.xlabel("Time (s)")
            plt.grid()
            plt.ylabel(" ")

    plt.show()


def getallbagfiles(datadirs):
    filenames = []
    for root, dirs, files in os.walk(datadirs):
        for filename in files:
            if filename.endswith(".bag"):
                filenames.append(os.path.join(root, filename))
    return filenames


def vector_to_mag(v):
    return np.linalg.norm(np.array([v.x, v.y, v.z]))


def plot(foldername):
    # matplotlib.rcParams['figure.figsize'] = [10, 10]
    matplotlib.rcParams['figure.figsize'] = [15, 4]
    bagfiles = getallbagfiles(foldername)
    for f in bagfiles:
        plot_results(f)
