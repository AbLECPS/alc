#!/usr/bin/env python2

'''
Generic ROS node for running an AssuranceMonitor in a standalone configuration.
If desired, code can be modified & integrated with existing LEC node for running in a combined LEC + AM configuration.

Author: Charlie Hartsell (charles.a.hartsell@vanderbilt.edu)
'''

import os
import rospy
import roslib.message
import message_filters
import std_msgs.msg
import alc_ros.msg
import alc_utils.common
import alc_utils.network_predictor
import json

DEFAULT_QUEUE_SIZE = 10
DEFAULT_TIME_SLOP_S = 0.1


class AssuranceMonitorNode:
    def __init__(self):
        # Test if any of the required parameters are missing
        req_param_labels = ['input_topic_type_pairs', 'lec_model_dir', 'confidence_output_topic']
        for label in req_param_labels:
            if not rospy.has_param('~%s' % label):
                raise rospy.ROSException('Required parameter missing, label=%s' % label)

        # Get parameters
        topic_type_pairs = rospy.get_param('~input_topic_type_pairs')
        self._lec_model_dir = rospy.get_param('~lec_model_dir')
        self._confidence_output_topic = rospy.get_param('~confidence_output_topic')
        self._assurance_monitor_path = rospy.get_param('~assurance_monitor_path',
                                                       os.path.join(self._lec_model_dir, "assurance_monitor.pkl"))
        self._subscriber_queue_size = rospy.get_param('~subscriber_queue_size', DEFAULT_QUEUE_SIZE)
        self._sync_time_slop = rospy.get_param('~sync_time_slop', DEFAULT_TIME_SLOP_S)

        # Derived parameters
        self._data_formatter_path = os.path.join(self._lec_model_dir, "data_formatter.py")
        lec_metadata = os.path.join(self._lec_model_dir, "model_metadata.json")
        with open(lec_metadata, 'r') as metadata_fp:
            metadata = json.load(metadata_fp)
        self._lec_input_shape = metadata["input_shape"]

        # Sanity checks on input parameters
        assert os.path.isdir(self._lec_model_dir), \
            'Specified LEC model directory {} is not a valid directory.'.format(self._lec_model_dir)
        assert os.path.isfile(self._assurance_monitor_path), \
            'Specified AssuranceMonitor path {} is not a valid file.'.format(self._assurance_monitor_path)
        assert os.path.isfile(self._data_formatter_path), \
            'Specified DataFormatter path {} is not a valid file.'.format(self._data_formatter_path)

        # Info messages
        rospy.loginfo("Using LEC model in directory %s" % self._lec_model_dir)

        # Load LEC
        self._lec_predictor = alc_utils.network_predictor.NetworkPredictor()
        self._lec_predictor.load_model(self._lec_model_dir,
                                       use_assurance=True,
                                       assurance_monitor_path=self._assurance_monitor_path)

        # Get message class from type string
        self._input_topic_type_pairs = []
        for topic_str, type_str in topic_type_pairs:
            message_type = roslib.message.get_message_class(type_str)
            if message_type is None:
                raise RuntimeError("Unable to identify message type '%s'" % type_str)
            else:
                self._input_topic_type_pairs.append((topic_str, message_type))

        # Subscribe to each of the input topics
        self._subscribers = []
        for topic_str, message_type in self._input_topic_type_pairs:
            rospy.loginfo("Subscribing to topic: %s" % topic_str)
            self._subscribers.append(message_filters.Subscriber(topic_str, message_type))

        # Setup ApproximateTimeSyncronizer
        self._time_sync = message_filters.ApproximateTimeSynchronizer(self._subscribers,
                                                                      queue_size=self._subscriber_queue_size,
                                                                      slop=self._sync_time_slop)
        self._time_sync.registerCallback(self.update_cb)

        # Setup confidence result publisher
        self._pub_desired_heading = rospy.Publisher(self._confidence_output_topic,
                                                    alc_ros.msg.AssuranceMonitorConfidenceStamped,
                                                    queue_size=1)
        rospy.loginfo("Publishing confidence output to topic: %s" % self._confidence_output_topic)

    def update_cb(self, *msgs):
        # Construct topic to message dictionary for NetworkPredictor
        topics_dict = {}
        for i, msg in enumerate(msgs):
            msg_topic, _ = self._input_topic_type_pairs[i]
            topics_dict[msg_topic] = msg

        # Run NetworkPredictor with assurance monitor
        _, _, assurance_result = self._lec_predictor.run(topics_dict)

        # Package assurance result into ROS messages and publish
        # Construct dimension and layout messages for each set of confidence values returned
        dim_msgs = []
        for i, confidence_values in enumerate(reversed(assurance_result)):
            if i > 0:
                last_dim_msg_stride = dim_msgs[i-1]
            else:
                last_dim_msg_stride = 1
            dim_msgs.append(std_msgs.msg.MultiArrayDimension("conf_%s" % i,
                                                             len(confidence_values),
                                                             len(confidence_values) * last_dim_msg_stride))
        layout_msg = std_msgs.msg.MultiArrayLayout(reversed(dim_msgs), 0)

        # Put confidence values into message
        confidence_values_flattened = []
        for confidence_values in assurance_result:
            confidence_values_flattened.extend(confidence_values)
        float_array_msg = std_msgs.msg.Float32MultiArray(layout_msg, confidence_values_flattened)
        conf_msg = alc_ros.msg.AssuranceMonitorConfidence(alc_ros.msg.AssuranceMonitorConfidence.NULL_TYPE,
                                                          [0.90, 0.95, 0.99],
                                                          float_array_msg)

        # Fill out header and publish
        stamped_conf_msg = alc_ros.msg.AssuranceMonitorConfidenceStamped()
        stamped_conf_msg.data = conf_msg
        stamped_conf_msg.header.frame_id = ""
        stamped_conf_msg.header.stamp = rospy.Time.now()
        self._pub_desired_heading.publish(stamped_conf_msg)


if __name__ == '__main__':
    print('Starting Assurance Monitor.')
    rospy.init_node('assurance_monitor_node.py')
    rospy.loginfo("Starting Assurance Monitor.")

    try:
        node = AssuranceMonitorNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        print('caught exception')
