# from keras import backend as K
import os
import numpy as np
import pCombination



class Config:
    def __init__(self, root_folder='.'):
        self.root_folder = root_folder
        self.window_size = [i for i in range(2, 11)]
        self.test_p_values = '{}/test_sequences_pvalues.pickle'.format(
            self.root_folder)
        self.performance_calibration_plots = '{}/performance_calibration/'.format(
            self.root_folder)
        self.credibility_hist_plots = '{}/Plots/credibility_histograms/'.format(
            self.root_folder)
        self.risk_coverage_plots = '{}/Plots/risk_coverage/'.format(
            self.root_folder)
        self.credibility_ecdf_plots = '{}/Plots/credibility_test/'.format(
            self.root_folder)
        self.check_and_create_path(self.performance_calibration_plots)
        self.check_and_create_path(self.credibility_hist_plots)
        self.check_and_create_path(self.risk_coverage_plots)
        self.check_and_create_path(self.credibility_ecdf_plots)

        self.combining_functions = {
            'merge': {
                'arith_avg': pCombination.arith_avg,
                'geom_avg': pCombination.geom_avg,
                'pmin': pCombination.pmin,
                'pmax': pCombination.pmax
            },
            'cdf': {
                'fisher': pCombination.ECF,
                'stouffer': pCombination.stouffer,
                'weighted_stouffer': pCombination.weighted_stouffer,
                'min_cdf': pCombination.min_cdf,
                'ord_stat_cdf': pCombination.ord_stat_cdf,
                'cauchi': pCombination.cauchi
            },
            'ecdf': {
                'ecdf_sum': pCombination.ecdf_sum,
                'ecdf_product': pCombination.ecdf_product,
                'ecdf_min': pCombination.pmin,
                'ecdf_max': pCombination.pmax,
                'ecdf_fisher': pCombination.ECF
            }
        }

    def check_and_create_path(self, folder_path):
        folder = os.path.join(self.root_folder, folder_path)
        if (not os.path.exists(folder)):
            os.makedirs(folder)
