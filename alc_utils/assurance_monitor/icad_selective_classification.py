#!/usr/bin/python
import os
import sys
import pickle
import collections
import numpy as np
from scipy import stats
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import tensorflow as tf
from selective_classification_config import Config as SConfig


class ICAD():

    def __init__(self, folder_path, num_classes, window_size=0):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        tf.get_logger().setLevel('INFO')

        physical_devices = tf.config.list_physical_devices('GPU')
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        except:
            pass

        self.num_classes = num_classes
        self.window_size = window_size

        self.model_path = folder_path
        self.classifier_model = None
        self.siamese_model = None
        self.calibration_nc = None
        self.centroids = None
        self.a_s = 0.9
        self.b_s = -0.3
        self.threshold_s = -0.145
        self.a_c = 0
        self.b_c = 0
        self.theshold_c = 0
        self.comb_type = None
        self.comb_function = None
        self.comb_am_config = None
        self.p_values = None
        self.use_sequence = False
        self.config = SConfig(self.model_path)
        self.best_correct_ratio_params = []
        self.best_no_decision_ratio_params = []

        print('model path', self.model_path)

        lec_weights_path = os.path.join(self.model_path, "classifier_model.h5")
        if (not os.path.exists(lec_weights_path)):
            lec_weights_path = os.path.join(
                self.model_path, "..", "classifier_model.h5")
        if (not os.path.exists(lec_weights_path)):
            lec_weights_path = os.path.join(
                self.model_path, "model_weights.h5")
        if (not os.path.exists(lec_weights_path)):
            lec_weights_path = os.path.join(
                self.model_path, "..", "model_weights.h5")

        nc_calibration_path = os.path.join(
            self.model_path, "calibration_nc_scores.pickle")
        siamese_model_path = os.path.join(self.model_path, "siamese_model.h5")
        snapshot_parameter_path = os.path.join(
            self.model_path, "snapshot.pickle")
        sequence_parameter_path = os.path.join(
            self.model_path, "sequence.pickle")

        msg = ""
        if (not os.path.exists(lec_weights_path)):
            msg += " LEC Weights"
        if (not os.path.exists(siamese_model_path)):
            msg += " Siamese Model"
        if (not os.path.exists(nc_calibration_path)):
            msg += " Calibration Score"
        if (not os.path.exists(snapshot_parameter_path)):
            msg += " a,b, threshold  values"

        if msg:
            complete_msg = 'Cannot load assurance monitor. Missing - '+msg
            raise ValueError(complete_msg)
        self.siamese_model = load_model(siamese_model_path)
        self.classifier_model = self.load_network(
            self.model_path, lec_weights_path)

        if (self.classifier_model is None):
            raise ValueError("Cannot load classifier network")

        with open(nc_calibration_path, "rb") as f:
            self.calibration_nc, self.centroids = pickle.load(f)

        if os.path.exists(snapshot_parameter_path):
            with open(snapshot_parameter_path, "rb") as f:
                self.a_s, self.b_s, self.threshold_s = pickle.load(f)
        else:
            print (
                'Cannot find snapshot_parameters.pickle. using default snapshot parameters')

        if os.path.exists(sequence_parameter_path):
            with open(sequence_parameter_path, "rb") as f:

                self.best_correct_ratio_params, self.best_no_decision_ratio_params = pickle.load(
                    f)
                print(self.best_correct_ratio_params)
                self.window_size = self.best_correct_ratio_params[0]
                self.comb_type = self.best_correct_ratio_params[1]
                self.comb_function = self.best_correct_ratio_params[2]
                self.a_c = self.best_correct_ratio_params[3]
                self.b_c = self.best_correct_ratio_params[4]
                self.threshold_c = self.best_correct_ratio_params[5]
                self.use_sequence = True
                self.load_sequence_configurations(self.model_path)
                # print(self.a_c)
                # print(self.b_c)
                # print(self.threshold_c)
        else:
            print ('Cannot find sequence_parameters.pickle. Not using sequence am')

        # Snapshot p vales:
        self.p_values = np.empty(self.num_classes)

        if (self.window_size):
            # self.p_value_window = = np.empty(self.window_size, self.num_classes)
            self.p_value_window = collections.deque(maxlen=self.window_size)
            self.conf_window = collections.deque(maxlen=self.window_size)
            self.cred_window = collections.deque(maxlen=self.window_size)
        else:
            self.p_value_window = None
            self.conf_window = None
            self.cred_window = None

        #self.a_s             = 0.9
        #self.b_s             = -0.3
        # self.threshold_s     = 0.6#-0.145
        # print(self.threshold_c)
        #self.threshold_c = 0.6

        sys.setrecursionlimit(40000)

    def load_network(self, model_path, lec_weights_path):
        import imp
        print('model_path ', model_path)
        network_path = os.path.join(model_path, "LECModel.py")

        if (not os.path.exists(network_path)):
            network_path = os.path.join(model_path, '..', 'LECModel.py')
        if (os.path.exists(network_path)):
            print('network_path ', network_path)
            if (os.path.exists(lec_weights_path)):
                self.netpath = network_path
                mods = imp.load_source('LEC_Model', network_path)
                if ('get_model' in dir(mods)):
                    net = mods.get_model()
                    net.load_weights(lec_weights_path)
                    print('loaded weights from {0}'.format(lec_weights_path))
                    return net
            else:
                print ('cannot load weights from {0}'.format(lec_weights_path))
        return None

    def load_sequence_configurations(self, model_path):
        import imp
        comb_am_config_path = os.path.join(model_path, "comb_am_config.py")
        print('comb_am_config_path', comb_am_config_path)
        if (not os.path.exists(comb_am_config_path)):
            comb_am_config_path = os.path.join(
                model_path, "..", "comb_am_config.py")
        if (os.path.exists(comb_am_config_path)):
            mods = imp.load_source('comb_am_config', comb_am_config_path)
            if ('Config' in dir(mods)):
                self.comb_am_config = mods.Config()

    def evaluate(self, lec_input):
        # if (self.thresholds == None):
        #    raise ValueError("Threshold value is not set")
        #print('thresholds {}, {}'.format(self.threshold_s,self.threshold_c))

        # compute embedding representation using the siamese network
        test_emb = self.siamese_model.predict(lec_input)
        prediction = np.argmax(self.classifier_model.predict(
            lec_input))  # LEC's classification
        softmax = np.max(prediction)  # LEC's classification softmax value

        p_values = np.empty(self.num_classes)
        centroid_distances = np.zeros(self.num_classes)
        for j in range(self.num_classes):
            centroid_distances[j] = np.linalg.norm(
                test_emb - self.centroids[j])
        for j in range(self.num_classes):
            temp_nc = centroid_distances[j] / float(
                np.min(centroid_distances[np.arange(len(centroid_distances)) != j]))
            p_values[j] = np.count_nonzero(self.calibration_nc >= temp_nc) / float(
                len(self.calibration_nc))  # Compute a p-value for each class
        # Credibility for the classification
        credibility = p_values[prediction]
        # Confidence for the classification
        confidence = 1 - \
            np.max(p_values[np.arange(len(p_values)) != prediction])

        decisions = {}
        decisions["comb"] = 0
        decisions["snapshot"] = 0

        self.p_values = p_values
        am_output = self.a_s * credibility + self.b_s * confidence

        #print("{}, {}, {}".format(self.a_s, self.b_s, self.threshold_s))

        if am_output >= self.threshold_s:  # Can we make a decision?
            decisions["snapshot"] = 1.0

        if (self.use_sequence):
            # Combining p values:

            combined_am_output = 0  # initial value

            self.p_value_window.append(p_values)
            self.conf_window.append(confidence)
            self.cred_window.append(credibility)

            # if sliding window is full
            #print (" {}, {} ".format(len(self.p_value_window),self.window_size))
            if len(self.p_value_window) == self.window_size:
                # Do the p value combination
                adjusted_p = self.config.combining_functions[self.comb_type][self.comb_function](
                    np.array(self.p_value_window))
                p_values_sort = np.sort(adjusted_p)
                # Compute credibility and confidence:
                prediction_credibility = p_values_sort[-1]
                prediction_confidence = p_values_sort[-1] - p_values_sort[-2]
                # Compute combined AM output
                combined_am_output = self.a_c * prediction_credibility + \
                    self.b_c * prediction_confidence
                decisions["comb"] = 1 if combined_am_output > self.threshold_c else 0
                # print(self.threshold_c)
                # print(combined_am_output)

        # print(decisions)

        return [self.p_values, prediction, credibility, confidence, decisions, am_output, softmax, combined_am_output]

    def get_p_values(self):
        return self.p_values

    def clear_windows(self):
        if (self.window_size):
            self.p_value_window.clear()
            self.conf_window.clear()
            self.cred_window.clear()

    def update_snapshot_params(self, **kwargs):
        self.threshold_s = kwargs.get('am_threshold', self.threshold_s)
        print('updated snapshot threshold {}'.format(self.threshold_s))

    def update_sequence_params(self, **kwargs):
        user_choice = kwargs.get('user_choice', 'trained_best')

        if (user_choice == 'trained_best'):
            return

        if (user_choice == 'override_threshold'):
            self.threshold_c = kwargs.get('am_s_threshold', self.threshold_c)
            print('updated sequence threshold {}'.format(self.threshold_c))
            return

        if (user_choice == 'override_all'):
            window_size = kwargs.get('window_size', self.window_size)
            comb_type = kwargs.get('comb_type', self.comb_type)
            comb_function = kwargs.get('comb_function', self.comb_function)
            found_trained_parameters = False
            if (self.comb_am_config and self.comb_am_config.coeffs):
                for coeff in self.comb_am_config.coeffs:
                    if window_size == coeff[0]:
                        if comb_type == coeff[1]:
                            if comb_function == coeff[2]:
                                self.a_c = coeff[3]
                                self.b_c = coeff[4]
                                self.threshold_c = coeff[5]
                                self.comb_type = comb_type
                                self.comb_function = comb_function
                                self.window_size = window_size
                                found_trained_parameters = True
                                break
            self.threshold_c = kwargs.get('am_s_threshold', self.threshold_c)

            if found_trained_parameter:
                print('overriding sequence parameters based on user choice')
            else:
                print(
                    'Could not find a,b values for user choice of window size, comb type, comb function. Reverting to best trained.')

            print('best trained type: {} function: {} window size: {}'.format(
                self.comb_type, self.comb_function, self.window_size))
            print('user choice type: {} function: {} window size: {}'.format(
                comb_type, comb_function, window_size))

        print('using sequence parameters a: {} b: {} threshold: {}'.format(
            self.a_c, self.b_c, self.threshold_c))
        del self.p_value_window
        del self.conf_window
        del self.cred_window

        self.p_value_window = collections.deque(maxlen=self.window_size)
        self.conf_window = collections.deque(maxlen=self.window_size)
        self.cred_window = collections.deque(maxlen=self.window_size)
