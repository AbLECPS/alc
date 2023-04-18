import numpy as np
import matplotlib.pyplot as plt
import math
import csv
import os
from sklearn import linear_model


class SelectiveClassification:
    def __init__(self, p_values, predictions, y_true):
        self.p_values = p_values
        self.p_values_sort = np.sort(self.p_values, axis=1)
        self.y_true = y_true
        self.y_pred = predictions
        self.credibility = np.array(
            [self.p_values[i, self.y_pred[i]] for i in range(len(y_true))])

        self.y_conf = np.array(
            [1-np.max(self.p_values[i, np.arange(p_values.shape[1]) != self.y_pred[i]]) for i in range(len(y_true))])
        self.incorrects = self.y_pred != self.y_true
        self.y_features = np.hstack(
            (self.credibility[:, np.newaxis], self.y_conf[:, np.newaxis]))

    def coverage(self, p_values, thres):
        return np.mean(p_values > thres)

    def risk(self, p_values, thres):
        return (np.dot(self.incorrects, (p_values > thres).astype(np.int))/(1.0*len(self.y_true)))/(1.0*self.coverage(p_values, thres))

    def aurc(self, p_values):
        s = self.risk(p_values, 0)
        count = 1
        for t in p_values:
            if math.isnan(self.risk(p_values, t)) == False:
                count += 1
                s += self.risk(p_values, t)
        return s/(1.0*count)

    def search_optimal(self):
        best_a = -2
        best_b = -2
        min_aurc = float("inf")
        for a in np.arange(-1, 1, 0.1):
            for b in np.arange(-1, 1, 0.1):
                if self.aurc(a*self.credibility+b*self.y_conf) < min_aurc:
                    min_aurc = self.aurc(a*self.credibility+b*self.y_conf)
                    best_a = a
                    best_b = b

        return best_a, best_b

    def rc_curve(self, p_values, plot_path, csv_path):
        if not os.path.exists(os.path.dirname(plot_path)):
            os.makedirs(os.path.dirname(plot_path))

        thresholds = np.sort(np.unique(p_values))

        r = [self.risk(p_values, thresholds[0]-0.5)]
        c = [self.coverage(p_values, thresholds[0]-0.5)]
        thr = [thresholds[0]-0.5]

        for t in thresholds[:-1]:
            r.append(self.risk(p_values, t))
            c.append(self.coverage(p_values, t))
            thr.append(t)

        file = open(csv_path, "w")
        writer = csv.writer(file)
        writer.writerow(['Threshold', 'Coverage', 'risk'])
        for i in range(len(r)):
            writer.writerow([thr[i], c[i], r[i]])
        file.close()
        print(thr)
        print(thr[5])
        return thr[5]
