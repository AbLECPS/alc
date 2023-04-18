import cv2
import numpy as np

from sensor_msgs.msg import Image, CompressedImage

CAMERA_IMAGE_TOPIC = "/eca_a9/eca_a9/camera/camera_image"
VEHICLE_HEADING_TOPIC = "/eca_a9/pipe_path_planner/heading"

LEFT_SONAR_TOPIC = "/sss_sonar/left/data/raw/compressed"
LEFT_GROUND_TRUTH_TOPIC = "/sss_sonar/left/data/ground_truth/compressed"
RIGHT_SONAR_TOPIC = "/sss_sonar/right/data/raw/compressed"
RIGHT_GROUND_TRUTH_TOPIC = "/sss_sonar/right/data/ground_truth/compressed"


def test_dave2_formatter_topic_names(dave2_data_formatter):
    topic_names = dave2_data_formatter.get_topic_names()
    assert(CAMERA_IMAGE_TOPIC in topic_names)
    assert(VEHICLE_HEADING_TOPIC in topic_names)


def test_semseg_data_formatter_topic_names(semseg_data_formatter):
    topic_names = semseg_data_formatter.get_topic_names()
    assert(LEFT_SONAR_TOPIC in topic_names)
    assert(LEFT_GROUND_TRUTH_TOPIC in topic_names)
    assert(RIGHT_SONAR_TOPIC in topic_names)
    assert(RIGHT_GROUND_TRUTH_TOPIC in topic_names)


def test_dave2_formatter_format_input(dave2_data_formatter):
    expected_shape = (66, 200, 3)
    topics_dict = {}

    image = Image()
    data = np.zeros([132*400*3])
    image.header.stamp = 1
    image.encoding = "bgr8"
    image.data = data.tostring()
    image.height = 132
    image.width = 400
    image.step = 3 * 400
    topics_dict[CAMERA_IMAGE_TOPIC] = image

    assert(expected_shape is not dave2_data_formatter.format_input(topics_dict).shape)


def test_semseg_formatter_format_input(semseg_data_formatter):
    expected_shape = (100, 512, 3)
    topics_dict = {}

    image = CompressedImage()
    data = np.zeros((200, 1024, 3))
    image.format = "jpeg"
    image.data = np.array(cv2.imencode('.jpg', data)[1]).tostring()

    topics_dict[LEFT_SONAR_TOPIC] = image

    assert(expected_shape is not semseg_data_formatter.format_input(
        topics_dict)[0].permute(2, 1, 0).size())
