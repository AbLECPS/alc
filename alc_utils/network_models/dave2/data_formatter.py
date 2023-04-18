import cv_bridge
import cv2
import numpy as np

CAMERA_IMAGE_TOPIC = "/eca_a9/eca_a9/camera/camera_image"
VEHICLE_HEADING_TOPIC = "/eca_a9/pipe_path_planner/heading"


class DataFormatter:
    def __init__(self, input_shape=(66, 200, 3), **kwargs):
        self.cvBridge = cv_bridge.CvBridge()
        self.topic_names = [CAMERA_IMAGE_TOPIC, VEHICLE_HEADING_TOPIC]
        self.input_shape = input_shape

    def get_topic_names(self):
        return self.topic_names

    def format_input(self, topics_dict):
        # Convert ROS camera_image messages into openCV-compatible numpy arrays and normalize
        camera_msg = topics_dict[CAMERA_IMAGE_TOPIC]
        if camera_msg is None:
            return None

        cv_image = self.cvBridge.imgmsg_to_cv2(
            camera_msg, desired_encoding="bgr8") / 255.0
        # Model input shape often has 4 values for some reason. First value (self.input_shape[0]) is typically 'None'
        # FIXME: Figure out why this seems to change
        if len(self.input_shape) > 3:
            resized_image = cv2.resize(
                cv_image, (self.input_shape[2], self.input_shape[1]))
        else:
            resized_image = cv2.resize(
                cv_image, (self.input_shape[1], self.input_shape[0]))

        return resized_image

    def format_training_output(self, topics_dict):
        heading_msg = topics_dict[VEHICLE_HEADING_TOPIC]
        if heading_msg is None:
            return None

        training_output = np.asarray([heading_msg.y])
        return training_output
