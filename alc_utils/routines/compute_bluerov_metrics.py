import sys
import os
import rospy
import rosbag
import matplotlib.pyplot as plt
import numpy as np
import tf
import math
import matplotlib
import re
from alc_utils.routines import plot_bluerov_results
from mpl_toolkits import mplot3d

alc_working_dir_env_var_name = "ALC_WORKING_DIR"
alc_working_dir_name = os.environ.get(alc_working_dir_env_var_name, None)

VEHICLE_ODOM_TOPIC = "/iver0/pose_gt_noisy_ned"
OBSTACLE_HSD_TOPIC = "/iver0/hsd_obstacle_avoidance"
FLS_RANGE_TOPIC = "/iver0/fls_echosunder"
LEC3_RANGE_TOPIC = "/lec3/fls_output"
VU_FLS_TOPIC = "/iver0/fls_lec"
NOISY_FLS_TOPIC = "/iver0/fls_output"
PIPE_DISTANCE_TOPIC = "/iver0/pipeline_distance_from_mapping"
LEC2_AM_L_TOPIC = "/iver0/cm_am/left"
LEC2_AM_R_TOPIC = "/iver0/cm_am/right"
# UUV_SPEED_TOPIC                    = "/iver0/speed"
UUV_SPEED_CMD_TOPIC = "/iver0/hsd_command"
BATT_LEVEL_TOPIC = "/iver0/pixhawk_hw"
CM_FAILSAFE_TOPIC = "/iver0/cm_failsafe"
CM_STATE_TOPIC = "/iver0/cm_state_machine"
DD_LEC = "/iver0/degradation_detector"
DEGRADATION_GT = "/iver0/degradation_gt"
DD_LEC_AM = "/lec_dd_am/p_value"
THRUSTERS_TOPIC = "/iver0/thruster_cmd_logging"
WP_COMPLETED_TOPIC = "/iver0/waypoints_completed"
XTRACK_TOPIC = "/iver0/xtrack_error"
OBSTACLE_TOPIC = "/iver0/obstacle_distances_gt"

DEFAULT_TOPICS = {"odom": VEHICLE_ODOM_TOPIC,
                  "obstacle_hsd":	OBSTACLE_HSD_TOPIC,
                  "fls_range":	FLS_RANGE_TOPIC,  # RAW
                  "lec3_range":	LEC3_RANGE_TOPIC,  
                  "vu_lec3_range":    VU_FLS_TOPIC,  # VU FLS3
                  "noisy_fls_range":  NOISY_FLS_TOPIC,  # RAW data with noise added
                  "pipe_distance":	PIPE_DISTANCE_TOPIC,
                  "lec2_am_l":	LEC2_AM_L_TOPIC,
                  "lec2_am_r":	LEC2_AM_R_TOPIC,
                  # "uuv_speed":	UUV_SPEED_TOPIC,
                  "uuv_speed_cmd":	UUV_SPEED_CMD_TOPIC,
                  "batt_level":	BATT_LEVEL_TOPIC,
                  "cm_failsafe":	CM_FAILSAFE_TOPIC,
                  "cm_state":	CM_STATE_TOPIC,
                  "dd_lec": DD_LEC,
                  "degradation_gt":	DEGRADATION_GT,
                  "dd_lec_am":	DD_LEC_AM,
                  "thrusters":    THRUSTERS_TOPIC}

CSV_DATA = {
    'GT degradation start [s]': 0,
    'GT degradation thr_id': 1,
    'GT degradation eff': 2,
    'Reallocation time [s]': 3,
    'FDI LEC thr_id': 4,
    'FDI LEC eff': 5,
    'Sim time [s]': 6,
    'Cross track error (wp) [m]': 7,
    'Time to complete (wp) [s]': 8,
    'Pipe avg distance [m]': 9,
    'Tracking error ratio': 10
}

line_dict = {
    '00_gt_deg_start': 16,
    '01_gt_deg_thr_id': 17,
    '02_gt_deg_eff': 18,
    '03_reallocate': 20,
    '04_lec_thr_id': 21,
    '05_lec_eff': 22,
    '10_simtime': 4,
    '21_xtrack': 13,
    '22_timecomplete': 14,
    '31_pipe_distance': 7,
    '32_tracking_error_ratio': 8
}


def fix_folder_path(folder_path):
    if (not folder_path):
        return None
    pos = folder_path.find('jupyter')
    if (pos == -1):
        return folder_path
    folder_path = folder_path[pos:]
    if (alc_working_dir_name):
        ret = os.path.join(alc_working_dir_name, folder_path)
        return ret
    return None


def parse_folder_list(folder_list):
    folders = []
    for f in folder_list:
        if isinstance(f, str):
            fixed_path = fix_folder_path(f)
            if (fixed_path):
                folders.append(fixed_path)
            continue
        if isinstance(f, dict):
            if (f.has_key('directory') and f.get('directory', None)):
                fixed_path = fix_folder_path(f['directory'])
                if (fixed_path):
                    folders.append(fixed_path)
            continue

    return folders


def getallevalfiles(datadirs):
    filenames = []
    for directory in datadirs:
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file == "bluerov_evaluation.txt":
                    filenames.append(os.path.join(root, file))
    return filenames


def getallbagfiles(datadirs):
    filenames = []
    for directory in datadirs:
        for root, dirs, files in os.walk(directory):
            for filename in files:
                # if filename.endswith(".bag"):
                if filename == ("recording.bag"):
                    filenames.append(os.path.join(root, filename))
    return filenames


def loadEval(filename):
    data_rows = []
    with open(filename) as fp:
        lines = fp.readlines()
        for key, value in sorted(line_dict.items()):
            val = re.split(' : | \[|\n| \n', lines[value])
            # data_rows.append([key, val[1]])
            # data_rows.append(key)
            data_rows.append(val[1])
    fp.close()
    data_rows.append(get_class(math.floor(
        float(data_rows[1])), float(data_rows[2])/100))
    return data_rows


def dump_csv(foldername, data):
    csv_filename = foldername + '_eval.csv'
    file = open(csv_filename, 'w')
    with file:
        write = csv.writer(file)
        write.writerows(data)
        print("\n> Eval data seved as: " + str(csv_filename))


def get_class(degraded_id, efficiency):
    num_classes = 22
    if degraded_id < 4:
        if efficiency <= 0.5:
            slot = 0
        elif efficiency <= 0.6:
            slot = 1
        elif efficiency <= 0.7:
            slot = 2
        elif efficiency <= 0.8:
            slot = 3
        elif efficiency <= 0.9:
            slot = 4
        else:
            slot = 5

        if slot < 5:
            class_val = slot + 5 * degraded_id
        else:
            class_val = num_classes - 1
    elif degraded_id < 6:
        if efficiency <= 0.9:
            class_val = 20 + (degraded_id - 4)
        else:
            class_val = num_classes - 1
    else:
        class_val = num_classes - 1
    return class_val


def parse_csv(foldername):
    files = getallevalfiles(foldername)
    sum_data = [
        'GT degradation start',
        'GT degradation thr_id',
        'GT degradation eff',
        'Reallocation time',
        'FDI LEC thr_id',
        'FDI LEC eff',
        'Sim time',
        'Avg. cross track error / wp',
        'Time to complete / wp',
        'Avg. pipe distance',
        'Tracking error ratio',
        'GT LEC class']
    for f in files:
        # print f
        data_rows = loadEval(f)
        if (not data_rows):
            print ('Failed ', f)
            continue
        sum_data = np.vstack((sum_data, data_rows))

    folder = str(Path(__file__).resolve().parent)
    foldername = folder.split(os.sep)[-1]
    dump_csv(foldername, sum_data)


def get_numeric_metrics(files):
    sum_data = []
    for f in files:
        # print f
        data_rows = loadEval(f)
        if (not data_rows):
            print ('Failed ', f)
            continue

        if len(sum_data) > 0:
            sum_data = np.vstack((sum_data, data_rows))
        else:
            sum_data = data_rows
    return sum_data


def run(folder_list=[], detailed=False):
    folders = parse_folder_list(folder_list)

    # Numeric data
    bag_files = getallbagfiles(folders)

    i = 0
    wp_finished = -1
    timestamp_ms = 0
    ros_start_time = rospy.Time(0)

    # DICTIONARY!!!!!!!!!!!!!!!!!!!!!!!
    heading_cmd = []
    xtrack_error = []
    mission_scores = []

    heading_score = []
    xtrack_score = []
    obstacle_score = []
    mission_scores = []
    ids = []

    min_obstacle_distances = []

    for filename in bag_files:
        bagfile = rosbag.Bag(filename)

        ids.append(i)
        print("\nPost process metrics - no.%s of %s:") % (i, len(bag_files) - 1)
        print("================================")
        print "Bagfile:\n%s" % (filename)

        # Get wp mission time:
        msgs = bagfile.read_messages(
            topics=[WP_COMPLETED_TOPIC], start_time=ros_start_time)
        if msgs is None:
            print "No data on topic %s present in bag file." % WP_COMPLETED_TOPIC
        else:
            for topic, msg, timestamp in msgs:
                timestamp_ms = (timestamp.secs * 1000.0) + \
                    (timestamp.nsecs / 1000000.0)
                if msg.data and wp_finished < 0:
                    wp_finished = timestamp_ms

        # Calculate obstacle distances
        msgs = bagfile.read_messages(
            topics=[OBSTACLE_TOPIC], start_time=ros_start_time)
        msg_count = bagfile.get_message_count(topic_filters=[OBSTACLE_TOPIC])
        # print(msg_count)
        min_obstacle_distance = None
        # if msgs is None:
        if msg_count == 0:
            print "No data on topic %s present in bag file." % OBSTACLE_TOPIC
        else:
            for topic, msg, timestamp in msgs:
                if wp_finished < timestamp_ms and len(msg.data) > 0:
                    if min_obstacle_distance is not None:
                        min_obstacle_distance = min(
                            np.min(msg.data), min_obstacle_distance)
                    else:
                        min_obstacle_distance = np.min(msg.data)
                    min_obstacle_distances.append(min_obstacle_distance)

        # Get HSD for angle error
        msgs = bagfile.read_messages(
            topics=[UUV_SPEED_CMD_TOPIC], start_time=ros_start_time)
        heading_cmd_sum = 0
        if msgs is None:
            print "No data on topic %s present in bag file." % UUV_SPEED_CMD_TOPIC
        else:
            heading_cmd_sum = 0
            for topic, msg, timestamp in msgs:
                if wp_finished < timestamp_ms:
                    heading_cmd_sum += abs(msg.heading)

        print "\n\t Simulation time: %s s" % (timestamp_ms//1000)
        print "\t------------------------------------------------"
        print "\t total angle error: \t\t%0.4f [deg]" % (heading_cmd_sum)
        print " \t avg.  angle error: \t\t%0.4f [deg]" % (
            heading_cmd_sum/timestamp_ms*1000)
        heading_cmd.append(heading_cmd_sum)

        # Get Cross track error
        msgs = bagfile.read_messages(
            topics=[XTRACK_TOPIC], start_time=ros_start_time)
        xtrack_sum = 0
        if msgs is None:
            print "No data on topic %s present in bag file." % XTRACK_TOPIC
        else:
            for topic, msg, timestamp in msgs:
                if wp_finished < timestamp_ms:
                    xtrack_sum += abs(msg.data)

        # Define maximum affordable limits:
        max_xtrack_error = 500
        max_angle_error = 30
        max_obstacle_distance = 50

        # Give max score if no obstacles present:
        if min_obstacle_distance is None:
            min_obstacle_distance = max_obstacle_distance

        # Calculate scores:
        # max 1.0 for heading error,
        # max 1.0 for cross track error,
        # max 1.0 for obstacle avoidance
        heading_score.append(
            1 - (min(heading_cmd_sum/timestamp_ms*1000, max_angle_error) / max_angle_error))
        xtrack_score.append(
            1 - (min(xtrack_sum/timestamp_ms*1000, max_xtrack_error) / max_xtrack_error))
        # Linear scoring:
        # obstacle_score.append(min(min_obstacle_distance, max_obstacle_distance) / max_obstacle_distance)
        # Nonlinear scoring:
        x = min(min_obstacle_distance, max_obstacle_distance) / \
            max_obstacle_distance
        obstacle_score.append(min(max(0.2932 * np.log(x) + 1.023, 0), 1))

        mission_score = heading_score[-1] + \
            xtrack_score[-1] + obstacle_score[-1]

        # Print output to Jupyter Notebook:
        print "\t total cross track error: \t%0.4f [m]" % (xtrack_sum)
        print "\t avg.  cross track error: \t%0.4f [m]" % (
            xtrack_sum/timestamp_ms*1000)
        print "\t minimum obstacle distance: \t%0.4f [m]" % (
            min_obstacle_distance)
        print "\t------------------------------------------------"
        print "\t heading score: \t\t%0.4f of 1.0" % (heading_score[-1])
        print "\t cross track score: \t\t%0.4f of 1.0" % (xtrack_score[-1])
        print "\t obstacle avoidance score: \t%0.4f of 1.0" % (
            obstacle_score[-1])
        print "\t------------------------------------------------"
        print "\t mission score: \t\t%0.4f of 3.0" % (mission_score)
        i += 1
        xtrack_error.append(xtrack_sum)
        mission_scores.append(mission_score)

    # 3D plots

    matplotlib.rcParams['figure.figsize'] = [15, 16]
    plt.figure(-5)
    ax = plt.axes(projection='3d')
    ax.scatter3D(obstacle_score, heading_score, xtrack_score, cmap='viridis')
    # ax.invert_xaxis()
    ax.invert_yaxis()
    # ax.invert_zaxis()
    ax.axis('equal')
    ax.set_xlabel("Obstacle avoidance score")
    ax.set_ylabel("Heading score")
    ax.set_zlabel("Cross track scor")
    plt.title("Mission scores", fontsize=14, fontweight='bold')
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_zlim(-0.1, 1.1)

    ax.view_init(25, 25)
    plt.show()

    # 1D plots

    start = 0
    end = len(heading_score)
    stepsize = 1.0  # For the sim 1D plot x axis

    matplotlib.rcParams['figure.figsize'] = [15, 4]

    plot_1d(-4, "Heading score", heading_score, start, end, stepsize)
    plot_1d(-3, "Cross track score", xtrack_score, start, end, stepsize)
    plot_1d(-2, "Obstacle avoidance score",
            obstacle_score, start, end, stepsize)

    # 2D plots

    plot_2d(id, "Heading score", heading_score,
            "Cross track core", xtrack_score)
    print("")

    print_statistics(
        heading_cmd, "Desired vs. actual heading difference (Heading cmd)[deg]:")

    print_statistics(
        xtrack_error, "Cross track error (distance from path)[m]:")

    print("Minimum obstace distance [m]:")
    if len(min_obstacle_distances) > 0:
        result_mean = np.nanmean(min_obstacle_distances)
        print(" - Arithmetic mean:")
        print("    " + str(result_mean) if result_mean >= 0 else "    n/a")
        print(" - Standard deviation:")
        print("    " + str(np.nanstd(min_obstacle_distances)))
        print(" - Variance:")
        print("    " + str(np.nanvar(min_obstacle_distances)))
    else:
        print("  --- No obstacles ---")
    print("")

    print_statistics(
        mission_scores, "\nOverall Scores based on post process metrics:")

    print_statistics(heading_score, "Mission heading scores:")

    print_statistics(xtrack_score, "Mission cross track scores:")

    print_statistics(obstacle_score, "Mission obstacle avoidance scores:")

    if detailed:

        print("\nStatistics of the realtime metrics:")
        print("=====================================")

        # Numeric data:

        # filenames = getallevalfiles(folders)
        # numeric_metrics = get_numeric_metrics(filenames)
        # print("Statistics:")
        # print("==========")
        # for metric in range(len(CSV_DATA)):
        #     print("- " + CSV_DATA.keys()[CSV_DATA.values().index(metric)] + ":")
        #     val = []
        #     for simualtion in range(len(filenames)):
        #         if numeric_metrics[simualtion][metric] != "-":
        #             val.append(float(numeric_metrics[simualtion][metric]))
        #         else:
        #             # pass
        #             val.append(-1.0)
        #     result_mean = np.nanmean(val)
        #     print(" o Arithmetic mean:")
        #     print("    " + str(result_mean) if result_mean >= 0 else "    n/a")
        #     print(" o Standard deviation:")
        #     print("    " + str(np.nanstd(val)))
        #     print(" o Variance:")
        #     print("    " + str(np.nanvar(val)))
        #     print("")

        # Plot data:
        bag_files = getallbagfiles(folders)
        matplotlib.rcParams['figure.figsize'] = [15, 4]
        print("\nOdometry:")
        print("-----------")
        i = 0
        for filename in bag_files:
            print("Simulation no.: " + str(i))
            plot_bluerov_results.plot_results(filename,
                                              topic_dict={
                                                  "odom":             VEHICLE_ODOM_TOPIC
                                              },
                                              start_time=0)
            i += 1

        print("\nForward Looking Sonar:")
        print("------------------------")
        i = 0
        for filename in bag_files:
            print("Simulation no.: " + str(i))
            plot_bluerov_results.plot_results(filename,
                                              topic_dict={
                                                  "fls_range":	    FLS_RANGE_TOPIC,  # RAW
                                                  "lec3_range":	    LEC3_RANGE_TOPIC,  
                                                  "vu_lec3_range":    VU_FLS_TOPIC,  # VU FLS3
                                                  "noisy_fls_range":  NOISY_FLS_TOPIC,  # RAW data with noise added
                                                  "pipe_distance":	PIPE_DISTANCE_TOPIC
                                              },
                                              start_time=0)
            i += 1

        print("\nSide Scan Sonar:")
        print("------------------")
        i = 0
        for filename in bag_files:
            print("Simulation no.: " + str(i))
            plot_bluerov_results.plot_results(filename,
                                              topic_dict={
                                                  "pipe_distance":	PIPE_DISTANCE_TOPIC
                                              },
                                              start_time=0)
            i += 1

        print("\nThrusters:")
        print("------------")
        matplotlib.rcParams['figure.figsize'] = [15, 16]

        i = 0
        for filename in bag_files:
            print("Simulation no.: " + str(i))
            plot_bluerov_results.plot_results(filename,
                                              topic_dict={
                                                  "thrusters":    THRUSTERS_TOPIC
                                              },
                                              start_time=0)
            i += 1


def print_statistics(var, text):
    print(text)
    print("----------------------------------")
    result_mean = np.nanmean(var)
    print(" - Arithmetic mean:")
    print("    " + str(result_mean) if result_mean >= 0 else "    n/a")
    print(" - Standard deviation:")
    print("    " + str(np.nanstd(var)))
    print(" - Variance:")
    print("    " + str(np.nanvar(var)))
    print("")


def plot_1d(id, label, y_values, start, end, stepsize):
    plt.figure(id)
    ax = plt.plot(y_values, 'o')
    plt.ylabel(label)
    plt.xlabel("Sim no.")
    plt.title(label, fontsize=14, fontweight='bold')
    plt.ylim(-0.1, 1.1)
    plt.grid()
    plt.xticks(np.arange(start, end, stepsize))


def plot_2d(id, x_label, x_data, y_label, y_data):
    plt.figure(-1)
    ax = plt.plot(x_data, y_data, 'o')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(x_label + " vs. " + y_label, fontsize=14, fontweight='bold')
    plt.ylim(-0.1, 1.1)
    plt.xlim(-0.1, 1.1)
    plt.grid()
    plt.show()
