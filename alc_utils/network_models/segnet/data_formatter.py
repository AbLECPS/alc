import torch
import cv_bridge
import cv2
import numpy as np
import scipy.misc

LEFT_SONAR_TOPIC = "/sss_sonar/left/data/raw/compressed"
LEFT_GROUND_TRUTH_TOPIC = "/sss_sonar/left/data/ground_truth/compressed"
RIGHT_SONAR_TOPIC = "/sss_sonar/right/data/raw/compressed"
RIGHT_GROUND_TRUTH_TOPIC = "/sss_sonar/right/data/ground_truth/compressed"


class DataFormatter:
    def __init__(self, input_shape=(100, 512), **kwargs):
        self.topic_names = [LEFT_SONAR_TOPIC, LEFT_GROUND_TRUTH_TOPIC,
                            RIGHT_SONAR_TOPIC, RIGHT_GROUND_TRUTH_TOPIC]
        self.bridge = cv_bridge.CvBridge()
        self.input_shape = input_shape

    def get_topic_names(self):
        return self.topic_names

    def format_input(self, topics_dict):
        formatted_inputs = []
        for sonar_topic in [LEFT_SONAR_TOPIC, RIGHT_SONAR_TOPIC]:
            # Convert sonar message to ndarray
            sonar_msg = topics_dict.get(sonar_topic, None)
            if sonar_msg is None:
                continue
            np_arr = np.fromstring(sonar_msg.data, np.uint8)
            # image_np = cv2.imdecode(np_arr, cv2.CV_LOAD_IMAGE_COLOR) # OpenCV < 3.0:
            image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # OpenCV >= 3.0:

            # FIXME: How to handle this if condition?
            # orig_size = image_np.shape[:-1]
            # if self.model_name in ["pspnet", "icnet", "icnetBN"]:
            #     # uint8 with RGB mode, resize width and height which are odd numbers
            #     image_np = scipy.misc.imresize(image_np, (orig_size[0] // 2 * 2 + 1, orig_size[1] // 2 * 2 + 1))
            # else:
            #     image_np = scipy.misc.imresize(image_np, (self.loader.img_size[0], self.loader.img_size[1]))

            # FIXME: Why are color channels reversed here?
            # resize image, reverse color channels (eg. RGB -> BGR), and cast to float.
            image_np = scipy.misc.imresize(
                image_np, (self.input_shape[0], self.input_shape[1]), interp="bicubic")
            image_np = image_np[:, :, ::-1]
            image_np = image_np.astype(np.float64)

            # FIXME: How to handle these?
            # image_np -= self.loader.mean
            # if self.img_norm:
            #     img = img.astype(float) / 255.0

            # Normalize
            image_np = image_np.astype(float) / 255.0

            # Height-Width-Channel -> Channel-Height-Width
            image_np = image_np.transpose(2, 0, 1)
            # image_np = np.expand_dims(image_np, 0)
            img = torch.from_numpy(image_np).float()

            formatted_inputs.append(img)

        return formatted_inputs

    def format_training_output(self, topics_dict):
        formatted_outputs = []
        for ground_truth_topic in [LEFT_GROUND_TRUTH_TOPIC, RIGHT_GROUND_TRUTH_TOPIC]:
            # Convert ROS Compressed Image message to numpy array
            gt_img = topics_dict.get(ground_truth_topic, None)
            if gt_img is None:
                continue
            np_arr = np.fromstring(gt_img.data, np.uint8)
            # image_np = cv2.imdecode(np_arr, cv2.CV_LOAD_IMAGE_COLOR) # OpenCV < 3.0:
            image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # OpenCV >= 3.0:

            # Translate segmented image from RGB to single-channel
            lbl = self._encode_segmap(image_np)

            # Resize image to match input shape
            lbl = lbl.astype(float)
            lbl = scipy.misc.imresize(
                lbl, (self.input_shape[0], self.input_shape[1]), "nearest", mode="F")
            lbl = lbl.astype(int)

            # # FIXME: Why are color channels reversed here?
            # # resize image, reverse color channels (eg. RGB -> BGR), and cast to float.
            # image_np = scipy.misc.imresize(image_np, (self.input_shape[0], self.input_shape[1]), interp="bicubic")
            # image_np = image_np[:, :, ::-1]
            # image_np = image_np.astype(np.float64)
            #
            # # FIXME: How to handle these?
            # # image_np -= self.loader.mean
            # # if self.img_norm:
            # #     img = img.astype(float) / 255.0
            #
            # # Height-Width-Channel -> Channel-Height-Width
            # image_np = image_np.transpose(2, 0, 1)
            # # image_np = np.expand_dims(image_np, 0)

            # Convert to PyTorch Tensor
            lbl = torch.from_numpy(lbl).long()
            formatted_outputs.append(lbl)

        return formatted_outputs

    @staticmethod
    def _encode_segmap(mask):
        mask = mask.astype(int)
        label_mask = np.zeros((mask.shape[0], mask.shape[1]))
        label_mask[mask[:, :, 0] == 255] = 1
        label_mask[mask[:, :, 1] == 255] = 2
        label_mask[mask[:, :, 2] == 255] = 3
        return np.array(label_mask, dtype=np.uint8)

    # def transform(self, img, lbl):
    #     img = img.astype(np.float32)
    #     img -= self.mean
    #     if self.img_norm:
    #         # Resize scales images from 0 to 255, thus we need
    #         # to divide by 255.0
    #         img = img.astype(float) / 255.0
    #     img = np.stack((img,img,img), axis=0)
    #     img = torch.from_numpy(img).float()
    #
    #     lbl = torch.from_numpy(lbl).long()
    #     return img, lbl

    def decode_output(self):
        pass
        # FIXME: No clue what "get prediction as argmax" is doing =
        #        The rest of this code seems to be data formatting (and confidence estimation)
        # # get prediction as argmax
        # pred = np.squeeze(outputs.data.max(1)[1].cpu().numpy(), axis=0)
        # if self.model_name in ["pspnet", "icnet", "icnetBN"]:
        #     pred = pred.astype(np.float32)
        #     # float32 with F mode, resize back to orig_size
        #     pred = scipy.misc.imresize(pred, orig_size, "nearest", mode="F")
        #
        # # get confidence map from probabilities
        # probs = F.softmax(outputs.detach()[0], dim=0).cpu().numpy()
        # confidence = np.zeros(pred.shape, dtype=np.float32)
        # for c in range(self.n_classes):
        #     confidence[pred == c] = probs[c][pred == c]
        #
        # decoded = self.loader.decode_segmap(pred)
        # logging.info("Classes found: " + str(np.unique(pred)))
        # return decoded, confidence
