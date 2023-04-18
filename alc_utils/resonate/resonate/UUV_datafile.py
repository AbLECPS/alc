import numpy as np
import rosbag
import pandas
import tf.transformations
import math
from datafile import Datafile

UUV_POSE_TOPIC = "/iver0/pose_gt"
#OBSTACLE_POSE_TOPIC = "/box2/pose_gt"
#OBSTACLE_POSE_TOPIC = "/iver0/box_position"
THRUSTER_DEGRADATION_TOPIC = "/iver0/degradation_gt"

# These values can be set manually according to UUV requirements.
UUV_RADIUS = 0.333
OBSTACLE_RADIUS = 0.5
FAR_ENCOUNTER_RANGE_M = 25.0
CLOSE_ENCOUNTER_RANGE_M = 25.0#5.0
COLLISION_RANGE_M = 25.0#2.0

MAX_MSG_TIME_SKEW_S = 0.1

# By default ten obstacle topics are published.
#OBSTACLE_TOPIC = ["/box1/pose_gt", "/box2/pose_gt", "/box3/pose_gt", "/box4/pose_gt",
#                  "/box5/pose_gt", "/box6/pose_gt", "/box7/pose_gt", "/box8/pose_gt",
#                  "/box9/pose_gt", "/box10/pose_gt"]

OBSTACLE_TOPIC = ["/iver0/box_position"]


class UUVDatafile(Datafile):
    def __init__(self):
        super(Datafile, self).__init__()
        self.independent_vars_disc = {}

    def calculate_separation_distance(self, uuv_pose_dataframe, obstacle_pose_dataframe):
        """Calculates the separation distance between obstacles and uuv
        from one bag file.
        Args:
            uuv_pose_dataframe: Dataframe containing position data of the uuv.
            obstacle_pose_dataframe: Dataframe containing position data of 
            obstacles.
        Returns:
            closest_approach: A min value from the separation distance of obstacles.
        """
        sep_dist = []
        uuv_pose = uuv_pose_dataframe
        obstacle_pose = obstacle_pose_dataframe
        for index, row in obstacle_pose.iterrows():
            # Center-Of-Mass (COM) and Point of Closest Approach (PCA)
            # For PCA, need to consider geometry of UUV and obstacle. Approximated as spheres here.
            com_dist = math.sqrt((row["x"] - pandas.to_numeric(uuv_pose["x"].iloc[index])) ** 2 +
                                 (row["y"] - pandas.to_numeric(uuv_pose["y"].iloc[index])) ** 2 +
                                 (row["z"] - pandas.to_numeric(uuv_pose["z"].iloc[index])) ** 2)
            pca_dist = com_dist - UUV_RADIUS - OBSTACLE_RADIUS
            sep_dist.append(pca_dist)
        sep_dist = np.array(sep_dist)
        closest_approach = np.min(sep_dist)
        # print(closest_approach)
        return closest_approach

    def read(self, filepath):
        """Overrides the read method of Datafile class to populate top_events
        consequences lists, and independent_vars_count dictionary.
        Args: 
            filepath: String of path address containing .bag file.
        """
        bag = rosbag.Bag(filepath)
        print(filepath)
        self.data = {}
        self.data["pose_gt"] = pandas.DataFrame(data=self.populate_uuv_pos_data(bag, UUV_POSE_TOPIC))
        self.req_topic = self.populate_obstacle_topic(bag)
        min_sep_dis = np.array([])
        # Compute minimum distance between UUV and all the dynamic obstacles.
        for i in range(0, len(self.req_topic)):
            # print("obstacle_pos{}".format(i))
            self.data["obstacle_pos{}".format(i)] = pandas.DataFrame(
                data=self.populate_obstacle_pos_data(bag, self.req_topic[i]))
            min_sep_dis = np.append(min_sep_dis, self.calculate_separation_distance(
                self.data["pose_gt"], self.data["obstacle_pos{}".format(i)]))
        # Add numpy array of minimum separation distance to the data dictionary.
        self.data["minimum_separation_distance"] = min_sep_dis

        # Get thruster degradation status
        self.data["thruster_efficiency"] = np.array(self.populate_thruster_status(bag, THRUSTER_DEGRADATION_TOPIC))
        thruster_degraded_indicies = self.data["thruster_efficiency"] < 1.0
        if np.count_nonzero(thruster_degraded_indicies) > 0:
            self.thruster_degradation_amount = 1 - np.average(
                self.data["thruster_efficiency"][thruster_degraded_indicies])
        else:
            self.thruster_degradation_amount = 0.0

        self.closest_approach = np.min(self.data["minimum_separation_distance"])
        self.closest_approach_index = np.argmin(self.data["minimum_separation_distance"])

        # Determine when an encounter has occurred (far and near)
        self.data["far_encounter"] = self.data["minimum_separation_distance"] < FAR_ENCOUNTER_RANGE_M
        print('far_encounter {0}'.format(self.data["far_encounter"]))
        self.data["close_encounter"] = self.data["minimum_separation_distance"] < CLOSE_ENCOUNTER_RANGE_M
        print('close_encounter {0}'.format(self.data["close_encounter"]))
        self.data["collision"] = self.data["minimum_separation_distance"] < COLLISION_RANGE_M
        print('collision {0}'.format(self.data["collision"]))

        # For convenience, store flags indicating if a threat, top event, or consequence has occurred
        self.threat_occurred = np.any(self.data["far_encounter"])
        self.top_occurred = np.any(self.data["close_encounter"])
        self.consequence_occurred = np.any(self.data["collision"])

        # eventually need to remove 'self' from rest of code above
        self.top_event = self.threat_occurred
        self.consequence = self.consequence_occurred
        self.independent_vars_cont = {
            "thruster_degredation": self.thruster_degradation_amount
        }

        # Close bag file after all data is read to save memory
        bag.close()

    def populate_uuv_pos_data(self, bag, uuv_pose_topic):
        """Reads through the rosbag of recording.bag file and stores 
        the Odometry message(i.e. x, y, z, and timestamp) in a dictionary.
        
        Args:
            bag: An object of Rosbag class. 
            uuv_pose_topic: String of uuv topic. 
        Returns:
            A dictionary containing Odometry data for the uuv.
        """
        # Read ground-truth position of vehicle
        pose_data = {"x": [], "y": [], "z": [], "orientation": [], "timestamp": []}
        for topic, msg, timestamp in bag.read_messages(UUV_POSE_TOPIC):
            pose_data["x"].append(msg.pose.pose.position.x)
            pose_data["y"].append(msg.pose.pose.position.y)
            pose_data["z"].append(msg.pose.pose.position.z)
            pose_data["orientation"].append(msg.pose.pose.orientation)
            pose_data["timestamp"].append(timestamp.to_sec())
        return pose_data

    def populate_obstacle_topic(self, bag):
        """Stores all the topics related to spawned obstacles in a bag file.
        
        Args:
            bag: An object of Rosbag class
        Returns:
            A list containing topics of all the obstacles.
        Raises:
            ValueError: If the bag file contains no obstacle topics.
        """
        req_topic = []
        for topic, msg, timestamp in bag.read_messages():
            if topic not in req_topic and topic in OBSTACLE_TOPIC:
                req_topic.append(topic)
        return req_topic

    def populate_obstacle_pos_data(self, bag, obstacle_pose_topic):
        """Reads through the  bag file and stores the obstacles Odometry message
        in a dictionary. Obstacle coordinates are relative to world reference
        frame. 
        Args:
            bag: An object of Rosbag class.
            obstacle_pose_topic: String of obstacle topic.
        Returns:
            A dictionary containing Odometry data for the obstacle.
        Raises:
            ValueError: If the coordinate frame of reference is not known.
        """
        obs_pos_data = {"x": [], "y": [], "z": [], "timestamp": []}
        for topic, msg, timestamp in bag.read_messages(obstacle_pose_topic):
            timestamp = timestamp.to_sec()
            # Obstacle coords are relative to world. Find vehicle position at
            # this timestamp (closest match).
            pose_timestamps = self.data["pose_gt"]["timestamp"]
            abs_time_diff = np.abs(pose_timestamps - timestamp)
            closest_match_idx = np.argmin(abs_time_diff)
            if abs_time_diff[closest_match_idx] > MAX_MSG_TIME_SKEW_S:
                raise ValueError("Closest messages exceed maximum allowed time skew.")
            closest_match_pose = self.data["pose_gt"].iloc[closest_match_idx]
            # Store obstacle coords if it's relative to world, otherwise perform
            # coordinate transformation.
            if msg.header.frame_id == "world":
                obs_pos_data["x"].append(msg.pose.pose.position.x)
                obs_pos_data["y"].append(msg.pose.pose.position.y)
                obs_pos_data["z"].append(msg.pose.pose.position.z)
                obs_pos_data["timestamp"].append(timestamp)
            elif msg.header.frame_id == "vehicle":
                quat_msg = closest_match_pose["orientation"]
                x_1 = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z, 0])
                q_1 = np.array([quat_msg.x, quat_msg.y, quat_msg.z, quat_msg.w])
                q_1_inv = tf.transformations.quaternion_inverse(q_1)
                x_0 = tf.transformations.quaternion_multiply(tf.transformations.quaternion_multiply(q_1, x_1), q_1_inv)
                obs_pos_data["x"].append(x_0[0] + closest_match_pose["x"])
                obs_pos_data["y"].append(x_0[1] + closest_match_pose["y"])
                obs_pos_data["z"].append(x_0[2] + closest_match_pose["z"])
                obs_pos_data["timestamp"].append(timestamp)
            else:
                raise ValueError("Unsupported coordinate reference frame.")
        return obs_pos_data

    def populate_thruster_status(self, bag, thruster_degradation_topic):
        """Reads through the bag file and stores thruster efficiency as a list.
        
        Args:
            bag: An object of Rosbag class.
            obstacle_pose_topic: String of thruster degradation topic.
        Returns:
            thruster_efficiency: A list of float containing thruster efficiency.
        """
        thruster_efficiency = []
        for topic, msg, timestamp in bag.read_messages(THRUSTER_DEGRADATION_TOPIC):
            self.thruster_id = msg.data[0]
            thruster_efficiency.append(msg.data[1])
        return thruster_efficiency
