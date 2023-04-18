"""
@author Feiyang Cai
@email feiyang.cai@vanderbilt.edu
@create date 2020-02-25 22:48:39
@modify date 2020-02-25 22:48:39
@desc [description]
"""


class StatefulDetector(object):
    def __init__(self, sigma, tau):
        self.S = 0
        self.sigma = sigma
        self.tau = tau

    def update_threshold(self, threshold):
        self.sigma = threshold[0]
        self.tau = threshold[1]

    def __call__(self, M):
        self.S = max(0.0, self.S+M-self.sigma)
        if self.S > self.tau:
            temp = self.S
            self.S = 0
            return temp, True
        else:
            return self.S, False


class StatelessDetector(object):
    def __init__(self, tau):
        self.tau = tau


    def update_threshold(self, threshold):
        self.tau = threshold[len(threshold)-1]

    def __call__(self, M):
        if M > self.tau:
            return True
        else:
            return False
