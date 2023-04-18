#!/usr/bin/python3
import scipy.integrate as integrate
from scipy import stats
import numpy as np
import queue
import random


class RPM(object):
    def __init__(self, epsilon, sliding_window_size=None):
        self.M = 1.0
        self.epsilon = epsilon
        self.sliding_window_size = sliding_window_size
        #self.sliding_window_size = 5
        if self.sliding_window_size:
            self.betting_queue = queue.Queue(self.sliding_window_size)

    def __call__(self, p):
        if (p <= 0.005):
            p = 0.005
        betting = self.epsilon * (p ** (self.epsilon-1.0))
        if self.sliding_window_size:
            if self.betting_queue.full():
                self.M /= self.betting_queue.get()
            self.betting_queue.put(betting)

        self.M *= betting
        return self.M


class SMM(object):
    def __init__(self, sliding_window_size=None):
        self.p_list = []
        self.sliding_window_size = sliding_window_size
        #self.sliding_window_size = 5

    def __integrand(self, x):
        result = 1.0
        for i in range(len(self.p_list)):
            result *= x*(self.p_list[i]**(x-1.0))
        return result

    def __call__(self, p):
        if (p <= 0.005):
            p = 0.005
        self.p_list.append(p)
        if self.sliding_window_size:
            if len(self.p_list) >= self.sliding_window_size:
                self.p_list = self.p_list[-1 * self.sliding_window_size:]
        M, _ = integrate.quad(self.__integrand, 0.0, 1.0)
        return M


class PIM(object):
    def __init__(self, sliding_window_size=None):
        # np.random.uniform(-1.0,2.0, 300).tolist()
        self.extended_p_list = [0.5, -0.5, 1.5]
        self.M = 1.0
        self.sliding_window_size = sliding_window_size

    def __call__(self, p):
        random.shuffle(self.extended_p_list)
        array = np.array(self.extended_p_list).reshape(1, -1)
        kernel = stats.gaussian_kde(array, bw_method='silverman')
        normalizer = kernel.integrate_box_1d(0.0, 1.0)
        betting = kernel(p)[0] / normalizer
        self.M *= betting
        self.extended_p_list.append(p)
        self.extended_p_list.append(-p)
        self.extended_p_list.append(2.0-p)
        if self.sliding_window_size:
            if len(self.extended_p_list) >= 3*self.sliding_window_size:
                self.extended_p_list = self.extended_p_list[-3 *
                                                            self.sliding_window_size:]
        return self.M
