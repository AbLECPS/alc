import sys
import os
import rospy
import rosbag
import rospkg
import matplotlib.pyplot as plt
import numpy as np
import tf
import math
import matplotlib
import re
import time
import test_comb_am_config
from alc_utils.routines import plot_bluerov_results
from mpl_toolkits import mplot3d
# workaround
rp = rospkg.RosPack()
sys.path.append(os.path.join(os.environ.get("ALC_HOME"),
                             "bluerov2/catkin_ws/src/vandy_bluerov/nodes"))
from dd_lec import DegradationDetectorLEC

import warnings
warnings.filterwarnings('ignore')

alc_working_dir_env_var_name = "ALC_WORKING_DIR"
alc_working_dir_name = os.environ.get(alc_working_dir_env_var_name, None)

DD_LEC = "/iver0/degradation_detector"
DEGRADATION_GT = "/iver0/degradation_gt"
DD_LEC_AM = "/lec_dd_am/p_value"
THRUSTERS_TOPIC = "/iver0/thruster_cmd_logging"
WP_COMPLETED_TOPIC = "/iver0/waypoints_completed"


DEFAULT_TOPICS = {"dd_lec": DD_LEC,
                  "degradation_gt":	DEGRADATION_GT,
                  "dd_lec_am":	DD_LEC_AM,
                  "thrusters":    THRUSTERS_TOPIC,
                  "wp_completed": WP_COMPLETED_TOPIC}


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


def run(folder_list=[], detailed=False, window_size=4, comb_type='merge', comFun='arith_avg', combined_am_threshold=0.40):
    start_time = time.time()

    folders = parse_folder_list(folder_list)

    # Numeric data
    bag_files = getallbagfiles(folders)
    # self.comb_type = comb_type
    # self.comFun = comFun
    normalizer = [1000, 1000, 1000, 1000, 1000,
                  1000, 6000, 6000, 6000, 6000, 6000, 6000, 30]
    # decision_threshold = 0.3
    num_classes = 22
    ann_input_len = 13
    rospy.set_param('model_dir', "jupyter/admin_BlueROV/LEC-DD/AM_final")
    # print(" \033[1;32m>>>>>>>>>>>>>>>>>>> DD (FDI) LEC LOADED <<<<<<<<<<<<<<<<<<<<<<<<<\033[0m ")

    # ids = []

    print("\nWindow, cType, cFun, Correct ratio, Incorrect ratio, No decision ratio")
    comb_config = test_comb_am_config.Config()
    for params in comb_config.coeffs:
        ddv2 = DegradationDetectorLEC(
            normalizer, params[5], num_classes, params[0])

        bag_id = 0
        wp_finished = -1
        timestamp_ms = 0
        ros_start_time = rospy.Time(0)
        incorrect_count = {
            "snapshot":     0,
            "comb":         0
        }
        correct_count = {
            "snapshot":     0,
            "comb":         0
        }

        no_decision_count = {
            "snapshot":     0,
            "comb":         0
        }

        decisions = {
            "snapshot":     False,
            "comb":         False
        }

        for filename in bag_files:
            bagfile = rosbag.Bag(filename)

            # ids.append(bag_id)

            """
            print("\nPost process metrics - no.%s of %s:") % (bag_id, len(bag_files) - 1)   
            print("================================")
            print "Bagfile:\n%s" % (filename)
            """

            gt_degraded_id = 6
            gt_degraded_efficiency = 1.00
            gt_degradation_start_time = -1

            # ddv2 = DegradationDetectorLEC(normalizer, decision_threshold, num_classes, window_size)
            # 10% faster this way:
            ddv2.clear_windows()

            reaction = {
                "snapshot":     False,
                "comb":         False
            }

            # Read the GT degradation data
            msgs = bagfile.read_messages(
                topics=[DEGRADATION_GT], start_time=ros_start_time)
            if msgs is None:
                print "No data on topic %s present in bag file." % DEGRADATION_GT
            else:
                for topic, msg, timestamp in msgs:
                    timestamp_ms = (timestamp.secs * 1000.0) + \
                        (timestamp.nsecs / 1000000.0)

                    # If state transitions from nominal to degraded
                    if (gt_degradation_start_time < 0 and msg.data[0] < 6):
                        gt_degradation_start_time = timestamp_ms
                        gt_degraded_id = int(msg.data[0])
                        gt_degraded_efficiency = msg.data[1]

            """
            print "Degradation starts at \t t= %ds" %(gt_degradation_start_time//1000)
            print "Degraded id \t%d" %(gt_degraded_id)
            print "Degraded efficiency \t %0.2f\n" %(gt_degraded_efficiency)
            """

            # Read the LEC input data, and give them to the LEC + AM(s)
            msgs = bagfile.read_messages(
                topics=[THRUSTERS_TOPIC], start_time=ros_start_time)
            if msgs is None:
                print "No data on topic %s present in bag file." % THRUSTERS_TOPIC
            else:
                for topic, msg, timestamp in msgs:
                    timestamp_ms = (timestamp.secs * 1000.0) + \
                        (timestamp.nsecs / 1000000.0)
                    if len(msg.data[0:ann_input_len]) == ann_input_len:
                        # Format LEC input
                        model_input = np.reshape(
                            msg.data[0:ann_input_len], (1, -1))

                        # print("\n\n\nModel input:\n"+str(model_input)+"\n\n\n")

                        # Run LEC prediction
                        # [prediction, credibility, confidence, decision, am_output, softmax, combined_am_output] = ddv2.predict(ddv2.normalize(model_input), comb_type, comFun)
                        [prediction, credibility, confidence, decision, am_output, softmax, combined_am_output] = ddv2.predict(
                            ddv2.normalize(model_input), params[1], params[2], params[3], params[4])
                        [degraded_id, efficiency] = ddv2.get_fault_from_class(
                            prediction)
                        p_values = ddv2.get_p_values()

                        decisions["comb"] = True if combined_am_output > combined_am_threshold else False
                        decisions["snapshot"] = True if decision == 1 else False

                        if timestamp_ms < gt_degradation_start_time:
                            uuv_state_str = "\033[1;32m GT Nom. (6|1.00|21)\033[0m"
                        else:
                            uuv_state_str = "\033[1;31m GT Deg. (" + "{:d}".format(gt_degraded_id) + "|" + "{:0.2f}".format(gt_degraded_efficiency) + \
                                "|" + str(get_class(gt_degraded_id,
                                                    gt_degraded_efficiency)) + ")\033[0m"

                            """
                            print("t=%ds, %s, lec:, %d, %d, %0.2f, am: %0.2f, decision: %d, comb_am: %0.2f, decision: %d" % (
                            timestamp.secs, uuv_state_str, prediction, degraded_id, efficiency, am_output, decision, combined_am_output, decisions["comb"]))          
                            """

                        # sys.stdout.write('.')
                        # sys.stdout.flush()

                        for am_type in decisions:
                            # UUV nominal and decision is made
                            if timestamp_ms < gt_degradation_start_time:
                                # AM assures a wrong LEC output -> Wrong reaction
                                if prediction < 20:
                                    if not reaction[am_type] and decisions[am_type]:
                                        reaction[am_type] = True
                                        incorrect_count[am_type] += 1
                                # Prediction for #20 and 21 does not leads to a decision, just warning msg
                            else:
                                # UUV degraded and decision is made
                                # Prediction is other than nominal or warning -> leads to reaction (reallocation)
                                if prediction < 20:
                                    if not reaction[am_type] and decisions[am_type]:
                                        reaction[am_type] = True
                                        if get_class(gt_degraded_id, gt_degraded_efficiency) != prediction:
                                            # Assured Output is incorrect -> Wrong reaction
                                            incorrect_count[am_type] += 1
                                        else:
                                            # Assured Output is correct -> Good reaction
                                            correct_count[am_type] += 1
                    else:
                        print("LEC input length error")

                for am_type in reaction:
                    if not reaction[am_type]:
                        no_decision_count[am_type] += 1

            bag_id += 1

        total_count = bag_id

        """
        print("\n\n\n#################################################################\n\n\n")
        print("Total simulation count: %d") % (bag_id)
        print("\nCombination AM:")
        print("Correct ratio: %0.2f") % (correct_count["comb"] / float(total_count))
        print("Incorrect ratio: %0.2f") % (incorrect_count["comb"] / float(total_count))
        print("No decision ratio: %0.2f") % (no_decision_count["comb"] / float(total_count))
        print("\nSnapshot AM:")
        print("Correct ratio: %0.2f") % (correct_count["snapshot"] / float(total_count))
        print("Incorrect ratio: %0.2f") % (incorrect_count["snapshot"] / float(total_count))
        print("No decision ratio: %0.2f") % (no_decision_count["snapshot"] / float(total_count))
        print("\nExecution time: %0.2f s" % (time.time() - start_time))
        """

        print("%d, %s, %s, %0.2f, %0.2f, %0.2f") % (
            params[0],
            params[1],
            params[2],
            correct_count["comb"] / float(total_count),
            incorrect_count["comb"] / float(total_count),
            no_decision_count["comb"] / float(total_count)
        )

    print("\n\n\n#################################################################\n\n\n")
    print("Total simulation count: %d") % (bag_id)
    print("\nSnapshot AM:")
    print("Correct ratio: %0.2f") % (
        correct_count["snapshot"] / float(total_count))
    print("Incorrect ratio: %0.2f") % (
        incorrect_count["snapshot"] / float(total_count))
    print("No decision ratio: %0.2f") % (
        no_decision_count["snapshot"] / float(total_count))
    print("\nExecution time: %0.2f s" % (time.time() - start_time))
