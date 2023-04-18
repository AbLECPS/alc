from __future__ import division
import matplotlib
matplotlib.use('Agg')
import time
import numpy as np
import os
from shutil import copyfile
from scipy import stats
import pickle
import sys
from collections import Counter
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.layers import Input, Lambda, Dense
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import Progbar
from tensorflow.keras.callbacks import TensorBoard
from sklearn import preprocessing
import random
import numpy.random as rng
import selective_classification_functions as selective_classification_functions
import matplotlib.pyplot as plt
import logging
from selective_classification_config import Config as SConfig
import time
import pandas as pd
import torch

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

def __randint_unequal(lower, upper):
    """
    Get two random integers that are not equal.

    Note: In some cases (such as there being only one sample of a class) there may be an endless loop here. This
    will only happen on fairly exotic datasets though. May have to address in future.
    :param lower: Lower limit inclusive of the random integer.
    :param upper: Upper limit inclusive of the random integer. Need to use -1 for random indices.
    :return: Tuple of (integer, integer)
    """
    int_1 = random.randint(lower, upper)
    int_2 = random.randint(lower, upper)
    while int_1 == int_2:
        int_1 = random.randint(lower, upper)
        int_2 = random.randint(lower, upper)
    return int_1, int_2


def class_separation(y_train_orig):
    class_idxs = []
    y_train = []
    for i in y_train_orig:
        y_train.append(np.argmax(i))

    for data_class in sorted(set(y_train)):
        class_idxs.append(np.where((y_train == data_class))[0])
    return class_idxs


def get_class(y_train_orig):
    y_train = []
    for i in y_train_orig:
        y_train.append(np.argmax(i))
    return np.array(y_train)


def euclidean_loss(y_true, y_pred):
    """
    Implementation of the triplet loss function
    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor data
            positive -- the encodings for the positive data (similar to anchor)
            negative -- the encodings for the negative data (different from anchor)
    Returns:
    loss -- real number, value of the loss
    """

    loss = y_true*K.square(y_pred)+(1-y_true)*K.square(K.maximum(5-y_pred, 0))

    return loss

from tensorflow.keras.preprocessing.sequence import pad_sequences

def get_batch(batch_size, x_train, y_train, idxs_per_class):
    """Create batch of n pairs, half same class, half different class"""
    n_classes = len(idxs_per_class)

    # randomly sample several classes to use in the batch
    categories = rng.choice(n_classes, size=(batch_size,), replace=True)
    # print("categories")
    # print(categories)

    # pairs1 has the anchors while pairs2 is either positive or negative
    pairs1 = []
    pairs2 = []

    # initialize vector for the targets
    targets = np.zeros((batch_size,), dtype='float')

    # make lower half of it '1's, so 2nd half of batch has same class
    targets[batch_size//2:] = 1.0
    for i in range(batch_size):
        category = categories[i]
        if i >= batch_size//2:  # positive
            idx = rng.choice(
                len(idxs_per_class[category]), size=(2,), replace=False)
            # print(idx[0],idx[1],y_train[idxs_per_class[category][idx[0]]],y_train[idxs_per_class[category][idx[1]]])
            pairs1.append(x_train[idxs_per_class[category][idx[0]]])
            pairs2.append(x_train[idxs_per_class[category][idx[1]]])
        else:  # negative
            category2 = (category+rng.randint(1, n_classes-1)
                         ) % n_classes  # pick from a different class
            # category2=(category+1)%2
            idx1 = rng.randint(0, len(idxs_per_class[category]))
            idx2 = rng.randint(0, len(idxs_per_class[category2]))
            # print(idx1,idx2,y_train[idxs_per_class[category][idx1]],y_train[idxs_per_class[category2][idx2]])
            pairs1.append(x_train[idxs_per_class[category][idx1]])
            pairs2.append(x_train[idxs_per_class[category2][idx2]])

    #padded1 = pad_sequences(pairs1)
    #padded2 = pad_sequences(pairs2)
    
    #X1 = np.expand_dims(padded1, axis = 0)
    #X2 = np.expand_dims(padded2, axis = 0)
    X1 = np.asarray(pairs1)
    X2 = np.asarray(pairs2)
    
    return X1,X2,targets


def write_log(callback, names, logs, batch_no):
    with callback.as_default():
        for name, value in zip(names, logs):
            tf.summary.scalar(name,value,step=batch_no)
            callback.flush()


def compute_ecdf(a):
    x, counts = np.unique(a, return_counts=True)
    cusum = np.cumsum(counts)
    return x, cusum / (1.0*cusum[-1])


def plot_ecdf(prediction_credibility, path):
    x, y = compute_ecdf(prediction_credibility)
    x = np.insert(x, 0, x[0])
    y = np.insert(y, 0, 0.)
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    fig = plt.figure(figsize=(7, 5))
    plt.plot(x, y, drawstyle='steps-post')
    plt.xlabel("Credibility")
    plt.ylabel("ecdf")
    plt.savefig(path)
    return x, y


def plot_credibility_hist(prediction_credibility, path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    fig = plt.figure(figsize=(7, 5))
    plt.hist(prediction_credibility, bins=100)
    plt.xlabel("Credibility")
    plt.ylabel("# of samples")
    plt.savefig(path)


def efficiency_calibration(p_values, y, epsilon):
    mult = 0
    error = 0
    for i in range(p_values.shape[0]):
        if np.count_nonzero(p_values[i] >= epsilon) > 1:
            mult += 1
        if p_values[i][y[i]] < epsilon:
            error += 1
    return mult/(1.0*p_values.shape[0]), error/(1.0*p_values.shape[0])


def plot_efficiency_calibration(p_values, y, e_start, e_step, e_end, path):
    plot_path = path+'.png'
    csv_path = path+'.csv'
    sig_levels = np.arange(e_start, e_end, e_step)
    perf_hist = np.empty(len(sig_levels))
    cal_hist = np.empty(len(sig_levels))
    for i, eps in enumerate(sig_levels):
        perf_hist[i], cal_hist[i] = efficiency_calibration(p_values, y, eps)

    fig = plt.figure(figsize=(6, 4))
    ax = plt.subplot()
    ax.grid(True, axis='both')
    ax.plot(sig_levels, perf_hist, label='Performance')
    ax.plot(sig_levels, cal_hist, label='Calibration')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel("Significance Level", fontsize=14)
    ax.set_ylabel("Performance & Calibration", fontsize=14)
    ax.legend(loc='upper right')
    fig.tight_layout()
    if not os.path.exists(os.path.dirname(plot_path)):
        os.makedirs(os.path.dirname(plot_path))
    fig.savefig(plot_path)

    d = {'Significance_Level': sig_levels,
         'Multiples': perf_hist, 'Error Rate': cal_hist}
    df = pd.DataFrame(d)
    df.to_csv(csv_path, index=False, header=True)
    return perf_hist, cal_hist


def plot_risk_coverage(p_values, y, e_start, e_step, e_end, path):
    plot_path = path+'.png'
    csv_path = path+'.csv'
    sig_levels = np.arange(e_start, e_end, e_step)

    coverage = []
    risk = []
    adjusted_p = p_values
    adjusted_labels = y
    stored_sig_levels = []

    for epsil in sig_levels:
        counter = 0
        correct = 0
        for i in range(len(adjusted_p)):
            if np.count_nonzero(adjusted_p[i] >= epsil) == 1:
                counter += 1
                if np.argmax(adjusted_p[i]) == adjusted_labels[i]:
                    correct += 1
        if counter > 0:
            coverage.append(counter/(1.0*len(adjusted_p)))
            risk.append(1-correct/(1.0*counter))
            stored_sig_levels.append(epsil)

    fig = plt.figure(figsize=(6, 4))
    ax = plt.subplot()
    ax.grid(True, axis='both')
    ax.scatter(coverage, risk, s=3)
    ax.set_xlabel("Coverage", fontsize=14)
    ax.set_ylabel("Risk", fontsize=14)
    ax.legend(loc='upper right')
    fig.tight_layout()
    if not os.path.exists(os.path.dirname(plot_path)):
        os.makedirs(os.path.dirname(plot_path))
    fig.savefig(plot_path)

    d = {'Risk': risk, 'Coverage': coverage, 'Epsilon:': stored_sig_levels}
    df = pd.DataFrame(d)
    df.to_csv(csv_path, index=False, header=True)


class selectiveClassification():
    def __init__(self, **kwargs):
        print("Initialize the selective classification am...")
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.dataset_training = None
        self.dataset_calibration = None
        self.dataset_testing = None
        self.calibration_NC = []
        self.siamese_net = None
        self.net = None
        self.model_path = None
        self.netpath = None
        self.callbacks = []
        self.kwargs = None
        self.a_s = None
        self.b_s = None
        self.threshold_s = None
        self.data_calibration_x = None
        self.data_calibration_y = None
        self.data_test_x = None
        self.data_test_y = None
        self.data_train_x = None
        self.data_train_y = None
        self.data_test_len_list = None
        self.get_params(**kwargs)

    def get_params(self, **kwargs):
        self.num_classes = kwargs.get('num_classes', 22)
        self.embeddings_size = kwargs.get('embeddings_size', 16)
        self.in_dims = kwargs.get('in_dims', 13)
        self.batch_size = kwargs.get('batch_size', 256)
        self.num_epochs = kwargs.get('num_epochs', 1000)
        self.epoch_length = kwargs.get('epoch_length', 200)
        self.kwargs = kwargs

    def siamese_model_train(self, save_path):

        physical_devices = tf.config.list_physical_devices('GPU')
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        except:
            pass

        #dataloader = self.dataloader_training
        #dataset = dataloader.dataset
        data_train_x = self.data_train_x
        data_train_y = self.data_train_y

        self.in_dims = data_train_x.shape[1]
        print ("in dims {0}".format(self.in_dims))
        idxs_per_class = class_separation(data_train_y)

        model = self.net  # .get_model(**self.kwargs)
        base_model = Model(inputs=model.input,
                           outputs=model.get_layer('embedding').output)
        base_model.summary()
        input_a = Input(shape=(self.in_dims,))
        input_b = Input(shape=(self.in_dims,))
        processed_a = base_model(input_a)
        processed_b = base_model(input_b)
        l2_distance_layer = Lambda(
            lambda tensors: K.sqrt(K.sum(K.square(tensors[0] - tensors[1]), axis=1, keepdims=True)))
        l2_distance = l2_distance_layer([processed_a, processed_b])
        self.siamese_net = Model([input_a, input_b], l2_distance)

        self.siamese_net.compile(
            loss=euclidean_loss, optimizer=Adam())#keras.optimizers.adam())
        self.siamese_net.summary()
        #self.siamese_net = self.siamese_net.to(self.device)

        log_path = os.path.join(self.model_path, 'logs')
        if not os.path.isdir(log_path):
            os.mkdir(log_path)

        callback = tf.summary.create_file_writer(log_path) # TensorBoard(log_path)
        #callback.set_model(self.siamese_net)

        best_loss = np.Inf
        train_step = 0
        losses = np.zeros(self.epoch_length)
        for epoch_num in range(self.num_epochs):
            print('Epoch {}/{}'.format(epoch_num+1, self.num_epochs))
            progbar = Progbar(
                self.epoch_length)   # keras progress bar
            iter_num = 0
            start_time = time.time()
            for batch_num in range(self.epoch_length):
                inputs1, inputs2, targets = get_batch(
                    self.batch_size , data_train_x, data_train_y, idxs_per_class)
                loss = self.siamese_net.train_on_batch(
                    x=(inputs1, inputs2), y=targets)
                write_log(callback, ['loss'], [loss, train_step], train_step)
                losses[iter_num] = loss
                iter_num += 1
                train_step += 1
                progbar.update(
                    iter_num, [('loss', np.mean(losses[:iter_num]))])

                if iter_num == self.epoch_length:
                    epoch_loss = np.mean(losses)
                    write_log(callback,
                              ['Elapsed_time', 'mean_loss'],
                              [time.time() - start_time, epoch_loss],
                              epoch_num)
                    if epoch_loss < best_loss:
                        print('Total loss decreased from {} to {}, saving weights'.format(
                            best_loss, epoch_loss))
                        best_loss = epoch_loss
                        base_model.save(save_path)

        return base_model

    def generate_embeddings(self, siamese_model_path):
        physical_devices = tf.config.list_physical_devices('GPU')
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        except:
            pass

        # Set learning phase to 0 for model.predict. Set to 1 for training
        K.set_learning_phase(0)

        siamese_model = load_model(siamese_model_path)
        data_train_x = self.data_train_x  # self.dataloader_training.dataset._input_data
        # self.dataloader_calibration.dataset._input_data
        data_calibration_x = self.data_calibration_x
        data_test_x = self.data_test_x  # self.dataloader_testing.dataset._input_data

        train_embeds = siamese_model.predict(data_train_x)
        calibration_embeds = siamese_model.predict(data_calibration_x)
        test_embeds = siamese_model.predict(data_test_x)

        print ('train_embeds {0}'.format(train_embeds))
        print ('calibration_embeds {0}'.format(calibration_embeds))
        print ('test_embeds {0}'.format(test_embeds))

        with open(os.path.join(self.model_path, "siamese_embeddings_train.pickle"), "wb") as f:
            pickle.dump(train_embeds, f)

        with open(os.path.join(self.model_path, "siamese_embeddings_validation.pickle"), "wb") as f:
            pickle.dump(calibration_embeds, f)

        with open(os.path.join(self.model_path, "siamese_embeddings_test.pickle"), "wb") as f:
            pickle.dump(test_embeds, f)

    def generate_classifier_probs(self):
        physical_devices = tf.config.list_physical_devices('GPU')
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        except:
            pass

        # Set learning phase to 0 for model.predict. Set to 1 for training
        K.set_learning_phase(0)
        sys.setrecursionlimit(40000)

        classifier_model = self.net
        data_test_x = self.data_test_x  # self.dataloader_testing.dataset._input_data
        probs = classifier_model.predict(data_test_x)
        with open(os.path.join(self.model_path, "classifier_probs_paths_test.pickle"), "wb") as f:
            pickle.dump(probs, f)

    def compute_nc_score(self):
        sys.setrecursionlimit(40000)

        with open(os.path.join(self.model_path, "siamese_embeddings_train.pickle"), "rb") as f:
            train_embeds = pickle.load(f)
        with open(os.path.join(self.model_path, "siamese_embeddings_validation.pickle"), "rb") as f:
            calibration_embeds = pickle.load(f)

        data_train_y = self.data_train_y  # self.dataloader_training.dataset._output_data
        # self.dataloader_calibration.dataset._output_data
        data_calibration_y = self.data_calibration_y
        data_test_y = self.data_test_y  # self.dataloader_testing.dataset._output_data

        data_train_y = get_class(data_train_y)
        data_calibration_y = get_class(data_calibration_y)
        data_test_y = get_class(data_test_y)

        # Nearest Centroid
        centroids = np.empty((self.num_classes, self.embeddings_size))
        print ('training embeds size {0}'.format(train_embeds.shape))
        print ('data_train_y size {0}'.format(data_train_y.shape))
        for i in range(self.num_classes):
            centroids[i] = np.mean(train_embeds[data_train_y == i], axis=0)

        calibration_nc = np.empty(len(data_calibration_y))
        temp_distances = np.zeros(self.num_classes)
        for i in range(len(data_calibration_y)):
            for j in range(self.num_classes):
                temp_distances[j] = np.linalg.norm(
                    calibration_embeds[i]-centroids[j])
            calibration_nc[i] = temp_distances[data_calibration_y[i]]/(1.0*np.min(
                temp_distances[np.arange(len(temp_distances)) != data_calibration_y[i]]))

        with open(os.path.join(self.model_path, "calibration_nc_scores.pickle"), "wb") as f:
            pickle.dump((calibration_nc, centroids), f, protocol=2)

    def compute_p_values(self):
        sys.setrecursionlimit(40000)

        with open(os.path.join(self.model_path, "siamese_embeddings_test.pickle"), "rb") as f:
            test_embeds = pickle.load(f)

        data_test_y = self.data_test_y  # self.dataloader_testing.dataset._output_data
        data_test_y = get_class(data_test_y)

        # Nearest Centroid
        p_values = np.empty((len(data_test_y), self.num_classes))
        with open(os.path.join(self.model_path, "calibration_nc_scores.pickle"), "rb") as f:
            calibration_nc, centroids = pickle.load(f)
        centroid_distances = np.zeros(self.num_classes)
        for i in range(len(data_test_y)):
            for j in range(self.num_classes):
                centroid_distances[j] = np.linalg.norm(
                    test_embeds[i]-centroids[j])
            for j in range(self.num_classes):
                temp_nc = centroid_distances[j]/(1.0*np.min(
                    centroid_distances[np.arange(len(centroid_distances)) != j]))
                p_values[i, j] = np.count_nonzero(
                    calibration_nc >= temp_nc)/(1.0*len(calibration_nc))

        with open(os.path.join(self.model_path, "test_p_values.pickle"), "wb") as f:
            pickle.dump(p_values, f)

    def selective_classification(self, C):
        sys.setrecursionlimit(40000)

        with open(os.path.join(self.model_path, "classifier_probs_paths_test.pickle"), "rb") as f:
            classification_probs = pickle.load(f)

        with open(os.path.join(self.model_path, "test_p_values.pickle"), "rb") as f:
            p_values = pickle.load(f)

        data_test_y = self.data_test_y  # self.dataloader_testing.dataset._output_data
        data_test_y = get_class(data_test_y)
        data_test_y = data_test_y.astype('int')
        p_values = p_values
        classification_probs = classification_probs

        sc = selective_classification_functions.SelectiveClassification(
            p_values, np.argmax(classification_probs, axis=1), data_test_y)
        sc.rc_curve(sc.credibility, os.path.join(self.model_path, "risk_coverage_credibility.png"),
                    os.path.join(self.model_path, "risk_coverage_credibility.csv"))
        print("AURC just credibility:", sc.aurc(sc.credibility))
        a, b = sc.search_optimal()
        print("Computed coefficients:", a, b)

        print("Improved AURC:", sc.aurc(a*sc.credibility+b*sc.y_conf))
        threshold = sc.rc_curve(a*sc.credibility+b*sc.y_conf, os.path.join(self.model_path, "risk_coverage_credibility_confidence.png"),
                                os.path.join(self.model_path, "risk_coverage_credibility_confidence.csv"))

        print(threshold)
        with open(os.path.join(self.model_path, "snapshot.pickle"), "wb") as f:
            pickle.dump((a, b, threshold), f)
        self.a_s = a
        self.b_s = b
        self.threshold_s = threshold

    def train_sequence(self, C):
        start_time = time.time()

        y_sim = self.data_test_y
        y_sim = get_class(y_sim)
        y_sim = y_sim.astype('int')

        test_sequence_length_list = self.data_test_len_list
        print('test sequence length list ' + str(test_sequence_length_list))

        with open(os.path.join(self.model_path, "test_p_values.pickle"), "rb") as f:
            p_values = pickle.load(f)

        print("Accuracy of highest p-value on single frames: {}".format(
            np.count_nonzero(y_sim == np.argmax(p_values, axis=1))/(1.0*len(y_sim))))
        plot_efficiency_calibration(
            p_values, y_sim, 0.01, 0.01, 1, C.performance_calibration_plots+'single_frame')

        p_values_sort = np.sort(p_values, axis=1)

        predictions = np.argmax(p_values, axis=1)
        prediction_credibility = p_values_sort[:, -1]
        correct = np.where(predictions == y_sim)[0]
        incorrect = np.where(predictions != y_sim)[0]
        logging.info(
            'Highest p-value accuracy: {:f}'.format(len(correct)/(1.0*len(y_sim))))
        plot_credibility_hist(prediction_credibility,
                              C.credibility_hist_plots+'single_frame.png')
        sc = selective_classification_functions.SelectiveClassification(
            p_values, np.argmax(p_values, axis=1), y_sim)
        print("AURC just credibility: ", sc.aurc(sc.credibility))
        a, b = sc.search_optimal()
        print("Computed coefficients: ", a, b)
        print("Improved AURC:", sc.aurc(a*sc.credibility+b*sc.y_conf))
        sc.rc_curve(a*sc.credibility+b*sc.y_conf, C.risk_coverage_plots +
                    '/single_window.png', C.risk_coverage_plots+'/single_window.csv')
        comb_am_path = os.path.join(self.model_path, "comb_am_config.py")

        file = open(comb_am_path, "w")
        file.write("#!/usr/bin/env python\n\n")
        file.write("class Config:\n")
        file.write("\tdef __init__(self):\n")
        file.write("\t\tself.coeffs = [\n")

        # TEST SEQUENCES
        best_correct_ratio = 0
        best_correct_ratio_params = []
        best_no_decision_ratio = 1
        best_no_decision_ratio_params = []

        # for window_size in C.window_size:
        #    for comb_type in ['merge','cdf']:
        #        for comFun in C.combining_functions[comb_type]:

        print(test_sequence_length_list)
        for window_size in [4, 6, 9]:
            for comb_type in ['merge', 'cdf']:
                for comFun in ['arith_avg', 'fisher']:
                    valid_comb_type_fns = list(
                        C.combining_functions[comb_type].keys())
                    print(valid_comb_type_fns)
                    if comFun not in valid_comb_type_fns:
                        print('skippinh {} {} '.format(comb_type, comFun))
                        continue
                    print('executing {} {} '.format(comb_type, comFun))
                    adjusted_p = []
                    adjusted_labels = []
                    start = 0
                    for test_sequence_length in test_sequence_length_list:
                        # slide the window through the whole sequence
                        for i in range(test_sequence_length-window_size+1):
                            # label of the first frame of the sliding window
                            s_label = y_sim[start + i+window_size-1]
                            adjusted_p.append(C.combining_functions[comb_type][comFun](
                                p_values[start+i:start+i+window_size, :]))
                            adjusted_labels.append(s_label)
                        start += test_sequence_length

                    adjusted_p = np.array(adjusted_p)
                    adjusted_labels = np.array(adjusted_labels)

                    plot_efficiency_calibration(adjusted_p, adjusted_labels, 0.01, 0.01, 1,
                                                C.performance_calibration_plots+'/window_size_{}/{}/{}'.format(window_size, comb_type, comFun))

                    p_values_sort = np.sort(adjusted_p, axis=1)
                    predictions = np.argmax(adjusted_p, axis=1)
                    prediction_credibility = p_values_sort[:, -1]
                    prediction_confidence = p_values_sort[:, -
                                                          1] - p_values_sort[:, -2]
                    # possibly for debug
                    correct = np.where(predictions == adjusted_labels)[0]
                    incorrect = np.where(predictions != adjusted_labels)[0]
                    print(
                        'Highest p-value accuracy: {:f}'.format(len(correct)/len(adjusted_labels)))
                    plot_credibility_hist(prediction_credibility, C.credibility_hist_plots +
                                          '/window_size_{}/{}/{}.png'.format(window_size, comb_type, comFun))
                    _, _ = plot_ecdf(prediction_credibility, C.credibility_ecdf_plots +
                                     '/window_size_{}/{}/{}.png'.format(window_size, comb_type, comFun))

                    plot_risk_coverage(adjusted_p, adjusted_labels, 0.001, 0.001, 0.1, C.risk_coverage_plots +
                                       '/window_size_{}/{}/{}'.format(window_size, comb_type, comFun))

                    sc = selective_classification_functions.SelectiveClassification(
                        adjusted_p, np.argmax(adjusted_p, axis=1), adjusted_labels)
                    # # print("AURC just credibility: ",sc.aurc(sc.credibility))
                    a, b = sc.search_optimal()
                    # # print("Computed coefficients: ",a,b)
                    # # print("Improved AURC:", sc.aurc(a*sc.credibility+b*sc.y_conf))

                    thr = sc.rc_curve(a*sc.credibility+b*sc.y_conf, C.risk_coverage_plots+'2/window_size_{}/{}/{}.png'.format(
                        window_size, comb_type, comFun), C.risk_coverage_plots+'2/window_size_{}/{}/{}.csv'.format(window_size, comb_type, comFun))
                    # # sc.logistic_regression()

                    combined_am_output = a * prediction_credibility + b * prediction_confidence
                    total_entries = len(combined_am_output)
                    num_decisions = 0
                    num_correct_decisions = 0
                    num_incorrect_decisions = 0
                    for idx in range(total_entries):
                        val = combined_am_output[idx]
                        pred = predictions[idx]
                        label = adjusted_labels[idx]
                        if (val > thr):
                            num_decisions += 1
                            if (pred == label):
                                num_correct_decisions += 1
                            else:
                                num_incorrect_decisions += 1
                    print ("total {} decisions {} correct {}  incorrect {}".format(
                        total_entries, num_decisions, num_correct_decisions, num_incorrect_decisions))

                    #log_str = str(window_size)+",'"+str(comb_type)+"','"+str(comFun)+"',"+str(a)+","+str(b)+","+str(thr)
                    correct_ratio = num_correct_decisions / \
                        float(total_entries)
                    incorrect_ratio = num_incorrect_decisions / \
                        float(total_entries)
                    no_decision_ratio = (
                        total_entries - num_decisions) / float(total_entries)

                    log_str = "%d, \'%s\', \'%s\', %0.4f, %0.4f, %0.4f, %0.4f, %0.4f, %0.4f" % (
                        window_size,
                        comb_type,
                        comFun,
                        a,
                        b,
                        thr,
                        correct_ratio,
                        incorrect_ratio,
                        no_decision_ratio
                    )

                    best_correct_ratio = 0
                    best_correct_ratio_params = []
                    best_no_decision_ratio = 1
                    best_no_decision_ratio_params = []
                    if (no_decision_ratio < best_no_decision_ratio) or (best_no_decision_ratio == 1):
                        best_no_decision_ratio = no_decision_ratio
                        best_no_decision_ratio_params = [
                            window_size, comb_type, comFun, a, b, thr, correct_ratio, no_decision_ratio]
                    elif (no_decision_ratio == best_no_decision_ratio):
                        if (correct_ratio > best_no_decision_ratio_params[6]):
                            best_no_decision_ratio_params = [
                                window_size, comb_type, comFun, a, b, thr, correct_ratio, no_decision_ratio]

                    if (best_correct_ratio < correct_ratio) or (best_correct_ratio == 0):
                        best_correct_ratio = correct_ratio
                        best_correct_ratio_params = [
                            window_size, comb_type, comFun, a, b, thr, correct_ratio, no_decision_ratio]
                    elif (correct_ratio == best_correct_ratio):
                        if (no_decision_ratio < best_correct_ratio_params[7]):
                            best_correct_ratio_params = [
                                window_size, comb_type, comFun, a, b, thr, correct_ratio, no_decision_ratio]

                    print(log_str)
                    # writer.writerow(['['+log_str+'],'])
                    file.write("\t\t\t[" + log_str + "],\n")

        file.write("\t\t]")
        file.close()
        print('Finished')
        print(str((time.time() - start_time)/60.0) + " minutes ---")
        with open(os.path.join(self.model_path, "sequence.pickle"), "wb") as f:
            pickle.dump((best_correct_ratio_params,
                         best_no_decision_ratio_params), f)

        return best_correct_ratio_params, best_no_decision_ratio_params

    def load_network(self, model_path):
        import imp
        print('model_path ', model_path)
        network_path = os.path.join(model_path, "LECModel.py")

        if (not os.path.exists(network_path)):
            network_path = os.path.join(model_path, '..', 'LECModel.py')
        if (os.path.exists(network_path)):
            print('network_path ', network_path)
            net_weights_path = os.path.join(model_path, 'model_weights.h5')
            if (not os.path.exists(net_weights_path)):
                net_weights_path = os.path.join(
                    model_path, '..', 'model_weights.h5')
            if (os.path.exists(net_weights_path)):
                self.netpath = network_path
                mods = imp.load_source('LEC_Model', network_path)
                if ('get_model' in dir(mods)):
                    # could there be other dimensions for the input-trained lec???,
                    # in any case, the lec function should accept **kwargs
                    #net_input_dict = {'num_classes':self.num_classes,'embeddings_size': self.embeddings_size,'in_dims':self.in_dims}
                    
                    self.net = mods.get_model()#**self.kwargs)
                    self.net.load_weights(net_weights_path)
                    print('loaded weights from {0}'.format(net_weights_path))

    def get_data(self):
        train_x = []
        train_y = []
        for batch_idx, (inputs, targets) in enumerate(self.dataloader_training):
            for j in range(len(inputs)):
                train_x.append(inputs[j].cpu().detach().numpy())
                train_y.append(targets[j].cpu().detach().numpy())

        self.data_train_x = np.array(train_x)
        self.data_train_y = np.array(train_y)

        print (' size of train x {0}'.format(self.data_train_x.shape))
        print (' size of train y {0}'.format(self.data_train_y.shape))

        train_x = []
        train_y = []
        for batch_idx, (inputs, targets) in enumerate(self.dataloader_calibration):

            for j in range(len(inputs)):
                train_x.append(inputs[j].cpu().detach().numpy())
                train_y.append(targets[j].cpu().detach().numpy())

        self.data_calibration_x = np.array(train_x)
        self.data_calibration_y = np.array(train_y)

        print (' size of calibration x {0}'.format(
            self.data_calibration_x.shape))
        print (' size of calibration y {0}'.format(
            self.data_calibration_y.shape))

        train_x = []
        train_y = []
        for batch_idx, (inputs, targets) in enumerate(self.dataloader_testing):

            for j in range(len(inputs)):
                train_x.append(inputs[j].cpu().detach().numpy())
                train_y.append(targets[j].cpu().detach().numpy())

        self.data_test_x = np.array(train_x)
        self.data_test_y = np.array(train_y)
        self.data_test_len_list = self.dataloader_testing.dataset._data_len_list

        print (' size of test x {0}'.format(self.data_test_x.shape))
        print (' size of test y {0}'.format(self.data_test_y.shape))

        # print('******************************')
        # data_test_class_y        = get_class(self.data_test_y)
        # print (' size of test y class {0}'.format(data_test_class_y.shape))
        # print(self.data_test_len_list)
        # for i in range(len(data_test_class_y)):
        #     print(data_test_class_y[i])
        # start = 0
        # for i in range(len(self.data_test_len_list)):
        #     stop = start + self.data_test_len_list[i]
        #     print('start = {}'.format(start))
        #     print('stop = {}'.format(stop))
        #     print('data = {}'.format(str(data_test_class_y[start:stop])))
        #     start = stop

    def fit(self, dataloader_training, dataloader_calibration, dataloader_testing, model_path, **kwargs):
        sys.setrecursionlimit(40000)
        self.model_path = model_path
        self.net = None
        self.load_network(model_path)
        if (not self.net):
            raise ValueError(
                'No LEC Network model definition found for training siamese network in assurance monitor training')

        self.dataloader_training = dataloader_training
        self.dataloader_calibration = dataloader_calibration
        self.dataloader_testing = dataloader_testing
        self.get_data()

        siamese_model_path = os.path.join(self.model_path, "siamese_model.h5")

        print(siamese_model_path)
        if (os.path.exists(siamese_model_path)):
            print ('siamese network exists.')

        if (self.net):
            if (not os.path.exists(siamese_model_path)):
                self.siamese_base_model = self.siamese_model_train(
                    siamese_model_path)

            self.generate_embeddings(siamese_model_path)
            self.generate_classifier_probs()
            self.compute_nc_score()
            self.compute_p_values()
            config = SConfig(self.model_path)
            self.selective_classification(config)
            self.train_sequence(config)

        return self.num_classes, self.a_s, self.b_s, self.threshold_s

    def save_model(self, path):
        pass
