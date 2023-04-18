from __future__ import print_function
import pandas as pd
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import datetime
import numpy as np
import shutil
import rosbag
import sys
import csv
import time
import string


"""util functions for plotting lec data"""
# define function to calculate the time from the ros stamps ['nsecs']


def processtime(nanoseconds):
    nanoseconds = nanoseconds.values.flatten()
    seconds = 0
    items = []
    for i in range(len(nanoseconds)):
        if(nanoseconds[i] == 0):
            seconds += 1
        tm = str(seconds)+"."+str(nanoseconds[i])
        items.append(float(tm))
    if(items[0] != 0.0):
        items = np.asarray(items)-items[0]
    else:
        items = np.asarray(items)
    return items

# define function to convert roll pitch values to x,y,z positions


def convert_rpy_to_xzy(pitch, yaw):
    pitch = pitch.values.flatten()
    yaw = yaw.values.flatten()
    x = []
    y = []
    z = []
    for i in range(0, len(pitch), 200):
        x_p = np.cos(yaw[i])*np.cos(pitch[i])
        y_p = np.sin(yaw[i])*np.cos(pitch[i])
        z_p = np.sin(pitch[i])
        vec = np.asarray([x_p, y_p, z_p])
        length = np.linalg.norm(vec)
        vec = (vec/length)
        x.append(vec[0])
        y.append(vec[1])
        z.append(vec[2])
    return np.asarray(x), np.asarray(y), np.asarray(z)

# calculate the norms for the pipes (closest distance to pipe vertex)


def calculate_norm_2d(pipeline_x, pipeline_y, uuv_x, uuv_y):
    norms = []
    for i in range(len(uuv_x)):
        uuv_pos = np.asarray([uuv_x[i], uuv_y[i]])
        # print(uuv_pos)
        distances = []
        for j in range(len(pipeline_x)):
            pipe_point = np.asarray([pipeline_x[j], pipeline_y[j]])
            diff = uuv_pos-pipe_point
            distance = np.linalg.norm(diff)
            distances.append(distance)
        norms.append(min(distances))
    return norms


def calculate_norm_3d(pipeline_x, pipeline_y, pipeline_z, uuv_x, uuv_y, uuv_z):
    norms = []
    for i in range(len(uuv_x)):
        uuv_pos = np.asarray([uuv_x[i], uuv_y[i], uuv_z[i]])
        distances = []
        for j in range(len(pipeline_x)):
            pipe_point = np.asarray(
                [pipeline_x[j], pipeline_y[j], pipeline_z[j]])
            diff = uuv_pos-pipe_point
            distance = np.linalg.norm(diff)
            distances.append(distance)
        norms.append(min(distances))
    return norms


def process_lec_topic(array):
    values = []
    for i in range(array.shape[0]):
        arr = np.fromstring(array[i][1:-1], sep=',')
        values.append(arr)
    return np.asarray(values)


def process_lec_times(seconds, nseconds):
    start_time = datetime.datetime.fromtimestamp(seconds[0])
    start_time_nsec = float("0."+str(nseconds[0]))
    times = [0]
    #print(start_time.strftime('%Y-%m-%d %H:%M:%S'),start_time_nsec)
    for i in range(1, len(seconds)):
        value = datetime.datetime.fromtimestamp(seconds[i])-start_time
        times.append(value.total_seconds() +
                     (float("0."+str(nseconds[i]))-start_time_nsec))
        # print(value.total_seconds())
    return times


def plot_segments_2d(x, y):
    segments = []
    for i in range(0, len(x), 2):
        seg_x = [x[i], x[i+1]]
        seg_y = [y[i], y[i+1]]
        segments.append([seg_y, seg_x])
        plt.plot(seg_y, seg_x, 'r')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


def generate_lec_csv(lec_pb, dir_name, pref_name):
    lec_inputs = process_lec_topic(lec_pb['state_vector'].values)
    lec_times = process_lec_times(
        lec_pb[["secs"]].values, lec_pb[["nsecs"]].values.flatten())
    csv_columns = [lec_times]
    for i in range(8):
        lec_input = np.asarray(lec_inputs[:, i])
        csv_columns.append(lec_input)
    csv_columns.append(np.asarray(lec_pb["heading"].values))
    csv_columns.append(np.asarray(lec_pb["speed"].values))

    csv_columns.append(np.asarray(lec_pb["x"].values))
    csv_columns.append(np.asarray(lec_pb["y"].values))
    csv_columns.append(np.asarray(lec_pb["z"].values))

    csv_columns.append(np.asarray(lec_pb["x.2"].values))
    csv_columns.append(np.asarray(lec_pb["y.2"].values))
    csv_columns.append(np.asarray(lec_pb["z.2"].values))

    csv_columns.append(np.asarray(lec_pb["x.3"].values))
    csv_columns.append(np.asarray(lec_pb["y.3"].values))
    csv_columns.append(np.asarray(lec_pb["z.3"].values))

    csv_columns.append(np.asarray(lec_pb["roll"].values))
    csv_columns.append(np.asarray(lec_pb["pitch"].values))
    csv_columns.append(np.asarray(lec_pb["yaw"].values))

    csv_columns.append(np.asarray(lec_pb["fin0_input"].values))
    csv_columns.append(np.asarray(lec_pb["fin0_output"].values))
    csv_columns.append(np.asarray(lec_pb["fin1_input"].values))
    csv_columns.append(np.asarray(lec_pb["fin1_output"].values))
    csv_columns.append(np.asarray(lec_pb["fin2_input"].values))
    csv_columns.append(np.asarray(lec_pb["fin2_output"].values))
    csv_columns.append(np.asarray(lec_pb["fin3_input"].values))
    csv_columns.append(np.asarray(lec_pb["fin3_output"].values))
    csv_columns.append(np.asarray(lec_pb["thrust_input"].values))
    csv_columns.append(np.asarray(lec_pb["thrust_output"].values))
    csv_columns.append(np.asarray(lec_pb['heading_change'].values))

    csv_columns = np.asarray(csv_columns)
    np.savetxt(os.path.join(dir_name, pref_name+"lec_inputs_outputs.csv"), csv_columns.T, delimiter=",",
               header="time,pipeline_orientation,pipe_range_port_side,pipe_range_stbd_side,time_since_last_detection,sas_range_port_side,sas_range_stbd_side,closest_point_of_approach,nearest_obstacle_range,heading,speed,x,y,z,vel_x,vel_y,vel_z,angular_x,angular_y,angular_z,roll,pitch,yaw,fin0_input,fin0_output,fin1_input,fin1_output,fin2_input,fin2_output,fin3_input,fin3_output,thrust_input,thrust_output,heading_change")
    return csv_columns


def plot_trajectory3d(feedback_db, pipeline_segments, obstacle_pd, plot_pipe=True):
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    x = feedback_db[['x']].values.flatten()
    y = feedback_db[['y']].values.flatten()
    z = feedback_db[['z']].values.flatten()
    pip_x = pipeline_segments[['x']].values.flatten()
    pip_y = pipeline_segments[['y']].values.flatten()
    pip_z = pipeline_segments[['z']].values.flatten()
    if plot_pipe:
        for i in range(0, len(pip_x), 2):
            seg_x = [pip_x[i], pip_x[i+1]]
            seg_y = [pip_y[i], pip_y[i+1]]
            seg_z = [pip_z[i], pip_z[i+1]]
            ax.plot3D(seg_y, seg_x, seg_z, '-r', label='pipeline')
    ax.plot3D(x, y, -z, 'b', label="vehicle trajectory")
    #ax.plot3D([-1885.24],[28.0173],[-46.5131],'ks',label="obstacle origin")
    if (obstacle_pd):
        for i in range(obstacle_pd.shape[0]):
            ax.plot3D([obstacle_pd.loc[i]['y']], [obstacle_pd.loc[i]['x']], [
                      obstacle_pd.loc[i]['z']], 'ks', label="obstacle centroid")

    ax.set_zlim3d(-70, 0)
    # ax.set_ylim3d(-200,200)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    plt.title("UUV and Pipe Position")
    plt.show()


def plot_lec_inputs(lec_db):
    time = lec_db['time']
    pipe_orientation = lec_db['pipeline_orientation']
    pipe_range_port_side = lec_db['pipe_range_port_side']
    pipe_range_stbd_side = lec_db['pipe_range_stbd_side']
    time_since_last_detection = lec_db['time_since_last_detection']
    sas_range_port_side = lec_db['sas_range_port_side']
    sas_range_stbd_side = lec_db['sas_range_stbd_side']
    closest_point_of_approach = lec_db['closest_point_of_approach']
    nearest_obstacle_range = lec_db['nearest_obstacle_range']
    heading = lec_db['heading']
    speed = lec_db['speed']

    plt.plot(time, pipe_orientation, 'r', label='pipeline orientation')
    plt.title("Lec input 1: Pipline orientation (degrees) vs time")
    plt.xlabel("time(s)")
    plt.ylabel("degrees")
    plt.legend()
    plt.show()

    plt.plot(time, pipe_range_port_side, 'r', label='pipe range port side')
    plt.title("Lec input 2: pipe range port side (meters) vs time")
    plt.xlabel("time(s)")
    plt.ylabel("meters")
    plt.legend()
    plt.show()

    plt.plot(time, pipe_range_stbd_side, 'r',
             label='pipe range starboard side')
    plt.title("Lec input 3: pipe range starboard side (meters) vs time")
    plt.xlabel("time(s)")
    plt.ylabel("meters")
    plt.legend()
    plt.show()

    plt.scatter(time, time_since_last_detection, marker='.',
                s=2.5, label='time since last detection')
    plt.title("Lec input 4: time_since_last_detection (s) vs time")
    plt.xlabel("time(s)")
    plt.ylabel("time since last detection (s)")
    plt.legend()
    plt.show()
    time_since_last_detection

    plt.plot(time, sas_range_port_side, 'r', label='sas range port side')
    plt.title("Lec input 5: sas range port side (resolution) vs time")
    plt.xlabel("time(s)")
    plt.ylabel("meters")
    plt.legend()
    plt.show()

    plt.plot(time, sas_range_stbd_side, 'r', label='sas range starboard side')
    plt.title("Lec input 6: sas range starboard side (resolution) vs time")
    plt.xlabel("time(s)")
    plt.ylabel("meters")
    plt.legend()
    plt.show()

    plt.plot(time, closest_point_of_approach, 'r',
             label='closest point of approach')
    plt.title("Lec input 7: closest point of approach (meters) vs time")
    plt.xlabel("time(s)")
    plt.ylabel("meters")
    plt.legend()
    plt.show()

    plt.plot(time, nearest_obstacle_range, 'r', label='nearest obstacle range')
    plt.title("Lec input 8: nearest obstacle range (meters) vs time")
    plt.xlabel("time(s)")
    plt.ylabel("meters")
    plt.legend()
    plt.show()

    plt.plot(time, heading, 'r', label='heading')
    plt.title("Lec output 1: heading (degrees) vs time")
    plt.xlabel("time(s)")
    plt.ylabel("degrees")
    plt.legend()
    plt.show()

    plt.plot(time, speed, 'r', label='speed')
    plt.title("Lec output 2: speed (m/s) vs time")
    plt.xlabel("time(s)")
    plt.ylabel("speed(m/s)")
    plt.legend()
    plt.show()


def fix_csv(file_name, output_name, dir_name, pref_name):

    fname = os.path.join(dir_name, pref_name + file_name)
    oname = os.path.join(dir_name, pref_name + output_name)

    from_file = open(fname)
    line = from_file.readline()
    new_line = line.replace("# ", "")

    to_file = open(oname, mode="w")
    to_file.write(new_line)
    shutil.copyfileobj(from_file, to_file)
    from_file.close()
    to_file.close()
    return oname


def generate_velocity_info_csv(velocity_pb, dir_name, pref_name):
    times = processtime(velocity_pb['nsecs'])
    csv_columns = [times]

    csv_columns.append(np.asarray(velocity_pb["vehicle_heading"].values))
    csv_columns.append(np.asarray(velocity_pb["vehicle_speed"].values))

    csv_columns.append(np.asarray(velocity_pb["track_heading"].values))
    csv_columns.append(np.asarray(velocity_pb['track_speed'].values))
    csv_columns.append(np.asarray(velocity_pb["fls_output"].values))

    csv_columns.append(np.asarray(velocity_pb["x"].values))
    csv_columns.append(np.asarray(velocity_pb["y"].values))
    csv_columns.append(np.asarray(velocity_pb["z"].values))

    csv_columns.append(np.asarray(velocity_pb["x.2"].values))
    csv_columns.append(np.asarray(velocity_pb["y.2"].values))
    csv_columns.append(np.asarray(velocity_pb["z.2"].values))

    csv_columns.append(np.asarray(velocity_pb["x.3"].values))
    csv_columns.append(np.asarray(velocity_pb["y.3"].values))
    csv_columns.append(np.asarray(velocity_pb["z.3"].values))

    csv_columns.append(np.asarray(velocity_pb["x.1"].values))
    csv_columns.append(np.asarray(velocity_pb["y.1"].values))
    csv_columns.append(np.asarray(velocity_pb["z.1"].values))
    csv_columns.append(np.asarray(velocity_pb["w"].values))

    csv_columns.append(np.asarray(velocity_pb["roll"].values))
    csv_columns.append(np.asarray(velocity_pb["pitch"].values))
    csv_columns.append(np.asarray(velocity_pb["yaw"].values))

    csv_columns.append(np.asarray(velocity_pb["fin0_input"].values))
    csv_columns.append(np.asarray(velocity_pb["fin0_output"].values))
    csv_columns.append(np.asarray(velocity_pb["fin1_input"].values))
    csv_columns.append(np.asarray(velocity_pb["fin1_output"].values))
    csv_columns.append(np.asarray(velocity_pb["fin2_input"].values))
    csv_columns.append(np.asarray(velocity_pb["fin2_output"].values))
    csv_columns.append(np.asarray(velocity_pb["fin3_input"].values))
    csv_columns.append(np.asarray(velocity_pb["fin3_output"].values))
    csv_columns.append(np.asarray(velocity_pb["thrust_input"].values))
    csv_columns.append(np.asarray(velocity_pb["thrust_output"].values))

    csv_columns = np.asarray(csv_columns)
    np.savetxt(os.path.join(dir_name, pref_name+"vehicle_data.csv"), csv_columns.T, delimiter=",",
               header="time,vehicle_heading,vehicle_speed,track_heading,track_speed,fls_output,x,y,z,vel_x,vel_y,vel_z,angular_x,angular_y,angular_z,quat_x,quat_y,quat_z,quat_w,roll,pitch,yaw,fin0_input,fin0_output,fin1_input,fin1_output,fin2_input,fin2_output,fin3_input,fin3_output,thrust_input,thrust_output")
    return csv_columns


def fix_feedback(dir_name, pref_name):
    fname = os.path.join(dir_name, pref_name +
                         "_slash_follow_lec_0_slash_feedback.csv")
    oname = os.path.join(dir_name, pref_name + "feedback.csv")
    frm_file = open(fname)
    to_file = open(oname, mode="w")
    fix_header = str(frm_file.readline())
    # There are two ways this is getting logged so I'll make sure they both get handled
    header_line = fix_header.replace(
        '"sas_range_stbd, closest_passing_distance, nearest_collision_object_range]",', "")
    header_line = header_line.replace(
        '- port_pipe_range,- stbd_pipe_range,- time_since_last_detection,- sas_range_port,- sas_range_stbd,- closest_passing_distance,- nearest_collision_object_range,', '')
    to_file.write(header_line)
    while fix_header:
        fix_header = frm_file.readline()
        # There are two ways this is getting logged so I'll make sure they both get handled
        line1 = str(fix_header).replace('"[pipe_heading, port_pipe_range, stbd_pipe_range, time_since_last_detection, sas_range_port,"',
                                        '"[pipe_heading,port_pipe_range,stbd_pipe_range,time_since_last_detection,sas_range_port,sas_range_stbd,closest_passing_distance,nearest_collision_object_range]"')
        line1 = line1.replace(
            '- pipe_heading', '"[pipe_heading,port_pipe_range,stbd_pipe_range,time_since_last_detection,sas_range_port,sas_range_stbd,closest_passing_distance,nearest_collision_object_range]"')
        if fix_header:
            to_file.write(line1)
    frm_file.close()
    to_file.close()
    from_file = open(oname)
    to_file = open(fname, mode="w")
    shutil.copyfileobj(from_file, to_file)
    os.remove(oname)
    to_file.close()
    from_file.close()


def fix_sysid_feedback(dir_name, pref_name):
    fname = os.path.join(dir_name, pref_name + "sys_id_lec1_1_hz.csv")
    oname = os.path.join(dir_name, pref_name + "lec1_mat_inputs.csv")
    frm_file = open(fname)
    to_file = open(oname, mode="w")
    fix_header = str(frm_file.readline())
    while fix_header:
        fix_header = frm_file.readline()
        if fix_header:
            to_file.write(fix_header)
    frm_file.close()
    to_file.close()


def fix_sysid_pipes(dir_name, pref_name):
    fname = os.path.join(dir_name, pref_name + "path_markers.csv")
    if (not os.path.exists(fname)):
        return
    oname = os.path.join(dir_name, pref_name + "pipe_mat_inputs.csv")
    frm_file = open(fname)
    to_file = open(oname, mode="w")
    fix_header = str(frm_file.readline())
    while fix_header:
        fix_header = frm_file.readline()
        if fix_header:
            to_file.write(fix_header)
    frm_file.close()
    to_file.close()


def fix_sysid_obstacles(dir_name, pref_name):
    fname = os.path.join(dir_name, pref_name + "obstacle_locations.csv")
    if (not os.path.exists(fname)):
        return
    oname = os.path.join(dir_name, pref_name + "obs_mat_inputs.csv")
    frm_file = open(fname)
    to_file = open(oname, mode="w")
    fix_header = str(frm_file.readline())
    while fix_header:
        fix_header = frm_file.readline()
        if fix_header:
            l1 = string.split(fix_header, ',')
            s1 = string.join(l1[1:], ',')
            to_file.write(s1)
    frm_file.close()
    to_file.close()


def process_bag(bag_name, dir_name, pref_name):
    bag = rosbag.Bag(bag_name)
    bagContents = bag.read_messages()

    # get list of topics from the bag
    listOfTopics = ["/follow_lec_0/feedback",
                    "/iver0/velocity_info", "/iver0/pose_gt_ned"]

    for topicName in listOfTopics:
        # Create a new CSV file for each topic
        fname = string.replace(topicName, '/', '_slash_') + '.csv'
        filename = os.path.join(dir_name, pref_name + fname)

        with open(filename, 'w+') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',')
            firstIteration = True  # allows header row
            # for each instant in time that has data for topicName
            for subtopic, msg, t in bag.read_messages(topicName):
                # parse data from this instant, which is of the form of multiple lines of "Name: value\n"
                # put it in the form of a list of 2-element lists
                msgString = str(msg)
                msgList = string.split(msgString, '\n')
                instantaneousListOfData = []
                for nameValuePair in msgList:
                    splitPair = string.split(nameValuePair, ':')
                    for i in range(len(splitPair)):  # should be 0 to 1
                        splitPair[i] = string.strip(splitPair[i])
                    instantaneousListOfData.append(splitPair)
                # write the first row from the first element of each pair
                if firstIteration:  # header
                    headers = ["rosbagTimestamp"]  # first column header
                    for pair in instantaneousListOfData:
                        headers.append(pair[0])
                    filewriter.writerow(headers)
                    firstIteration = False
                # write the value from each pair to the file
                values = [str(t)]  # first column will have rosbag timestamp
                for pair in instantaneousListOfData:
                    if len(pair) > 1:
                        values.append(pair[1])
                filewriter.writerow(values)
    bag.close()

# function to generate the csvs


def generate_sys_id_csvs(dir_name, pref_name, genstatus):

    if (genstatus):
        oname = os.path.join(dir_name, pref_name + 'sys_id_lec1_1_hz.csv')
        if (os.path.exists(oname)):
            fix_sysid_pipes(dir_name, pref_name)
            fix_sysid_obstacles(dir_name, pref_name)
            return oname

    velocity_info = os.path.join(
        dir_name, pref_name+'_slash_iver0_slash_velocity_info.csv')
    velocity_info_data = pd.read_csv(velocity_info)
    velocity_info_data = velocity_info_data[velocity_info_data.columns[~velocity_info_data.isnull(
    ).all()]]
    generate_velocity_info_csv(velocity_info_data, dir_name, pref_name)
    fix_csv('vehicle_data.csv', 'vehicle_data_20_hz.csv', dir_name, pref_name)
    os.remove(os.path.join(dir_name, pref_name+'vehicle_data.csv'))

    feedback_data_lec = os.path.join(
        dir_name, pref_name+'_slash_follow_lec_0_slash_feedback.csv')
    feedback_lec_data = pd.read_csv(feedback_data_lec)
    feedback_lec_data = feedback_lec_data[feedback_lec_data.columns[~feedback_lec_data.isnull(
    ).all()]]
    generate_lec_csv(feedback_lec_data, dir_name, pref_name)
    lec_csv_filename = fix_csv(
        'lec_inputs_outputs.csv', 'sys_id_lec1_1_hz.csv', dir_name, pref_name)
    os.remove(os.path.join(dir_name, pref_name+'lec_inputs_outputs.csv'))

    fix_sysid_feedback(dir_name, pref_name)
    fix_csv("path_markers.csv", "path_markers.csv", dir_name, pref_name)
    fix_csv("path_markers_segments.csv",
            "path_markers_segments.csv", dir_name, pref_name)
    fix_sysid_pipes(dir_name, pref_name)
    fix_sysid_obstacles(dir_name, pref_name)

    writestatusfile(dir_name, pref_name)

    return lec_csv_filename


def writestatusfile(dir_name, pref_name):
    fname = os.path.join(dir_name, pref_name+'_csvs')
    to_file = open(fname, mode="w")
    to_file.write('done')
    to_file.close()


def getallbagfiles(datadirs):
    filenames = []
    for root, dirs, files in os.walk(datadirs):
        for file in files:
            if file.endswith(".bag"):
                filenames.append(os.path.join(root, file))
    return filenames


def getallsysidfiles(datadirs):
    filenames = []
    for root, dirs, files in os.walk(datadirs):
        for file in files:
            if file.endswith("lec1_mat_inputs.csv"):
                filenames.append(os.path.join(root, file))
    return filenames


def getallpipefiles(datadirs):
    filenames = []
    for root, dirs, files in os.walk(datadirs):
        for file in files:
            if file.endswith("pipe_mat_inputs.csv"):
                filenames.append(os.path.join(root, file))
    return filenames


def getallobstaclefiles(datadirs):
    filenames = []
    for root, dirs, files in os.walk(datadirs):
        for file in files:
            if file.endswith("obs_mat_inputs.csv"):
                filenames.append(os.path.join(root, file))
    return filenames


def getBaseName(bagfilename):
    dir_name = os.path.dirname(bagfilename)
    base_name = os.path.basename(bagfilename)
    base_prefix = os.path.splitext(base_name)[0]
    base_file_path = os.path.join(dir_name, base_prefix)
    return base_file_path


def getAllBaseNames(folder):
    ret = []
    for f in folders:
        bag_files = getallbagfiles(f)
        for b in bag_files:
            base_name = getBaseName(b)
            ret.append(base_name)
    return ret


def checkPrevSysIDGen(base_name):
    fname = base_name + '_csvs'
    if (os.path.exists(fname)):
        return True
    return False


def generateSysIDData(folders):
    ret = []
    ret_base = []
    for f in folders:
        bag_files = getallbagfiles(f)
        for b in bag_files:
            dir_name = os.path.dirname(b)
            file_name = os.path.basename(b)
            pref_name = os.path.splitext(file_name)[0]
            base_name = os.path.join(dir_name, pref_name)
            check_prev_gen = checkPrevSysIDGen(base_name)
            if (not check_prev_gen):
                process_bag(b, dir_name, pref_name)
                fix_feedback(dir_name, pref_name)
            ret.append(generate_sys_id_csvs(
                dir_name, pref_name, check_prev_gen))

            ret_base.append(base_name)
    return ret, ret_base


def plot_pipe_segment(base_name):
    pipeline_segments = base_name + 'path_markers_segments.csv'
    pipeline_segments_data = pd.read_csv(pipeline_segments)
    pipe_seg_x = pipeline_segments_data['x']
    pipe_seg_y = pipeline_segments_data['y']
    pipe_seg_Z = pipeline_segments_data['z']
    segments = plot_segments_2d(pipe_seg_y, pipe_seg_x)


def plot_3d_trajectory(base_name):
    pipeline_segments = base_name + 'path_markers_segments.csv'
    pipeline_segments_data = pd.read_csv(pipeline_segments)

    feedback_data_lec = base_name + '_slash_follow_lec_0_slash_feedback.csv'
    feedback_lec_data = pd.read_csv(feedback_data_lec)
    feedback_lec_data = feedback_lec_data[feedback_lec_data.columns[~feedback_lec_data.isnull(
    ).all()]]
    lec_times = process_lec_times(
        feedback_lec_data[["secs"]].values, feedback_lec_data[["nsecs"]].values.flatten())

    obstacle_locations = base_name + 'obstacle_locations.csv'
    obstacle_locations_data = None
    if(os.path.exists(obstacle_locations)):
        obstacle_locations_data = pd.read_csv(obstacle_locations)

    plot_trajectory3d(feedback_lec_data, pipeline_segments_data,
                      obstacle_locations_data, True)


def get_3d_trajectory_data(base_name):
    pipeline_segments = base_name + 'path_markers_segments.csv'
    pipeline_segments_data = pd.read_csv(pipeline_segments)

    feedback_data_lec = base_name + '_slash_follow_lec_0_slash_feedback.csv'
    feedback_lec_data = pd.read_csv(feedback_data_lec)
    feedback_lec_data = feedback_lec_data[feedback_lec_data.columns[~feedback_lec_data.isnull(
    ).all()]]
    lec_times = process_lec_times(
        feedback_lec_data[["secs"]].values, feedback_lec_data[["nsecs"]].values.flatten())

    obstacle_locations = base_name + 'obstacle_locations.csv'
    obstacle_locations_data = None
    if(os.path.exists(obstacle_locations)):
        obstacle_locations_data = pd.read_csv(obstacle_locations)

    return feedback_lec_data, pipeline_segments_data, obstacle_locations_data


def plot_lec_data(base_name):
    file_name = base_name + 'sys_id_lec1_1_hz.csv'
    data = pd.read_csv(file_name)
    data.columns.tolist()
    plot_lec_inputs(data)
    return data


def get_vehicle_data(base_name):
    file_name = base_name + 'vehicle_data_20_hz.csv'
    data = pd.read_csv(file_name)
    return data
