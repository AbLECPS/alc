import numpy as np
import glob
import os
import function_fitting
from UUV_datafile import UUVDatafile
from CARLA_datafile import CARLADatafile
import argparse
from argparse import RawTextHelpFormatter
from tqdm import tqdm

class CollatedData(object):
    def __init__(self):
        self.top_events = []  # list of booleans
        self.consequences = []  # list of booleans
        self.list_independent_vars_cont = {}  # dictionary, string -> list of floats
        self.list_independent_vars_disc = {}  # dictionary, string -> list of ints

    def build(self, list_of_data_objs):
        """TODO: Add comment here."""
        self.list_independent_vars_cont = {}
        for data in list_of_data_objs:
            self.top_events.append(data.top_event)
            self.consequences.append(data.consequence)
            if (data.independent_vars_cont):
                for var_name, var_value in data.independent_vars_cont.items():
                    if self.list_independent_vars_cont.get(var_name, False):
                        self.list_independent_vars_cont[var_name].append(var_value)
                    else:
                        self.list_independent_vars_cont[var_name] = [var_value]
            if (data.independent_vars_disc):
                for var_name, var_value in data.independent_vars_disc.items():
                    if self.list_independent_vars_disc.get(var_name, False):
                        self.list_independent_vars_disc[var_name].append(var_value)
                    else:
                        self.list_independent_vars_disc[var_name] = [var_value]

    def get_run_info(self, run_index):
        """TODO: ADD COMMENT HERE!"""
        dt = Datafile()
        dt.top_event = self.top_events[run_index]
        dt.consequence = self.consequences[run_index]
        dt.independent_vars_cont = {}
        dt.independent_vars_disc = {}
        for key, value in self.list_independent_vars_cont.items():
            dt.independent_vars_cont[key] = value[run_index]
        for key, value in self.list_independent_vars_disc.items():
            dt.independent_vars_disc[key] = value[run_index]
        return dt

    def show(self):
        """TODO: ADD COMMENT HERE!"""
        print("Top events:", self.top_events)
        print("Consequences:", self.consequences)
        for key, value in self.list_independent_vars_cont.items():
            print("{}: {}".format(key, value))
        for key, value in self.list_independent_vars_disc.items():
            print("{}: {}".format(key, value))
        print('')


# Produces the stats and table for all disc. variables
def disc_anaylsis(collated_data):
    """TODO: ADD Comment to this method for posterity."""

    num_cases = len(collated_data.list_independent_vars_disc) + 1
    consequence_count = np.zeros(num_cases)
    top_event_count = np.zeros(num_cases)
    trial_count = np.zeros(num_cases)  # array of the size of indep var, each index would be a different var
    total_trials = 0  # or even better, make an object
    total_top = 0
    total_consequence = 0

    for trial in range(len(collated_data.top_events)):
        check_for_no_disc = 0
        for disc_var in range(num_cases - 1):
            # check if the discrete variable occurred
            if collated_data.list_independent_vars_disc.values()[disc_var][trial]:
                # disc var 1, i.e. the first one defined in the dict, will relate to the first index in the arrays.
                check_for_no_disc += 1
                trial_count[disc_var] += 1
                if collated_data.consequences[trial]:
                    consequence_count[disc_var] += 1
                if collated_data.top_events[trial]:
                    top_event_count[disc_var] += 1

        # at this point i should check to see if no disc vars happened in this trail
        # last index, i = num_cases-1, is reserved for no disc_var
        if check_for_no_disc == 0:  # i.e. no disc var occurred in this trial
            # disc var 1, i.e. the first one defined in the dict, will relate to the first index in the arrays.
            trial_count[num_cases - 1] += 1
            if collated_data.consequences[trial]:
                consequence_count[num_cases - 1] += 1
            if collated_data.top_events[trial]:
                top_event_count[num_cases - 1] += 1

        total_trials += 1
        if collated_data.consequences[trial]:
            total_consequence += 1
        if collated_data.top_events[trial]:
            total_top += 1

    # Print results
    print("Variable\t\tTrials\tTE\tTE%\tP|FM_EST  P|!FM_EST\tC\tC%")
    print("---------------------------------------------------------------------------------------")

    list_disc_var_names = collated_data.list_independent_vars_disc.keys()
    # print(list_disc_var_names)

    for disc_var in range(num_cases):  # todo make sure this range is correct
        if trial_count[disc_var] > 0:
            top_event_pct = 100 * top_event_count[disc_var] / float(trial_count[disc_var])
            consequence_pct = 100 * consequence_count[disc_var] / float(trial_count[disc_var])
        else:
            top_event_pct = 0.0
            consequence_pct = 0.0

        prob_success_est = (trial_count[disc_var] - top_event_count[disc_var] + 1) / float(trial_count[disc_var] + 2)
        p_not_est = 1 - (total_top - top_event_count[disc_var] + 1) / float(total_trials - trial_count[disc_var] + 2)
        if disc_var != num_cases - 1:
            print("%.15s\t\t%d\t%d\t%.2f\t%.2f\t  %.2f  \t%d\t%.2f" % (
                list_disc_var_names[disc_var], trial_count[disc_var], top_event_count[disc_var], top_event_pct,
                prob_success_est, p_not_est, consequence_count[disc_var], consequence_pct))
        else:
            print("%.15s\t\t%d\t%d\t%.2f\t%.2f\t  %.2f  \t%d\t%.2f" % (
                "No disc var", trial_count[disc_var], top_event_count[disc_var], top_event_pct,
                prob_success_est, p_not_est, consequence_count[disc_var], consequence_pct))

    # Print totals (excluding the no-fault special case) as well
    print("---------------------------------------------------------------------------------------")
    total_top_pct = 100 * total_top / total_trials
    total_consequence_pct = 100 * total_consequence / total_trials
    total_prob_success = (total_trials - total_top + 1) / float(total_trials + 2)
    print("%.15s\t\t\t%d\t%d\t%.2f\t%.2f      %d\t%.2f\n" % ("TOTALS", total_trials, total_top, total_top_pct,
                                                             total_prob_success, total_consequence,
                                                             total_consequence_pct))


# Produces stats and plots for each cont. variables
def cont_analysis(cd):
    """TODO: ADD COMMENT HERE!"""
    # Calculate some simple statistics
    avg_top_prob = ((np.sum(cd.top_events) + 1) / float(len(cd.top_events) + 2))
    avg_col_prob = ((np.sum(cd.consequences) + 1) / float(len(cd.consequences) + 2))
    avg_col_prob_after_top = ((np.sum(cd.consequences) + 1) / float(np.sum(cd.top_events) + 2))
    
    print ("\n\nAverage Probability computed for the events....")
    print("  TOP-Event Probability        : %f" % avg_top_prob)
    print("  Consequence-Event Probability: %f" % avg_col_prob)
    print("  Consequence Probability given Top-Event occurred: %f" % avg_col_prob_after_top)
    print("\n\n")

    

    results = {}
    results['top_probability']=avg_top_prob
    results['consequence_probability']=avg_col_prob
    results['conditional_probability']=avg_col_prob_after_top

    for key, values in cd.list_independent_vars_cont.items():

        values_np = np.array(values)
        top_np = np.array(cd.top_events)
        cons_np = np.array(cd.consequences)

        # # Fit a conditional binomial distribution to the data with max-likelihood for the TOP event
        x_range = np.linspace(np.min(values), np.max(values), 100)
        fit_results = None
        print('Estimated parameters to predict likelihood of the event occurence....\n')
        print('Top Event:')
        print('------------\n')
        try:
            fit_results, opposite_sig_max_likelihood_coeff_top_event, results['top_fit'] = function_fitting.max_likelihood_conditional_fit(values_np, top_np)
        except ValueError as e:
            print("Exception occurred when fitting sigmoid for TOP event:\n%s" % str(e))
        else:
            y_sig_max_likelihood = function_fitting.bounded_sigmoid(x_range, *fit_results.x)
            adjusted_likelihood = y_sig_max_likelihood / avg_top_prob
            #print("TOP Sigmoid Fit: ", fit_results)
            #print("\n\n")
            
        
        # Fit a conditional binomial distribution to the data with max-likelihood for the TOP event
        fit_results = None
        print('\nConsequence Event:')
        print('-------------------\n')
        try:
            fit_results, opposite_sig_max_likelihood_coeff_conseq, results['consequence_fit'] = function_fitting.max_likelihood_conditional_fit(values_np, cons_np)
        except ValueError as e:
            print("Exception occurred when fitting sigmoid for Collision event:\n%s" % str(e))
        else:
            collision_y_sigmoid = function_fitting.bounded_sigmoid(x_range, *fit_results.x)
            collision_adj_likelihood = collision_y_sigmoid / avg_col_prob
            #print("Collision Sigmoid Fit: ", fit_results)
            #print("\n\n")

        
        import json
        with open('results.json', 'w') as f:
            json.dump(results, f, indent=4, sort_keys=True)


def calculate_stats(datafiles):
    """Plot the outcome of each scenario against the average assurance monitor value
    Args:
        
        """
    cd = CollatedData()
    cd.build(datafiles)
    # cd.show()

    # Datasets with no top events or collisions can cause issues with max likelihood estimation. Print warning.
    if np.sum(cd.top_events) == 0:
        print("WARNING: No Top events occurred in the provided datasets.")
    if np.sum(cd.consequences) == 0:
        print("WARNING: No Consequence events occurred in the provided datasets.")

    if cd.list_independent_vars_disc:
        disc_anaylsis(cd)

    if cd.list_independent_vars_cont:
        cont_analysis(cd)


def run_general_analyze_script(recent_execution, df_class=UUVDatafile, file_path=None, top_folders=["/"]):
    # FIXME: This needs some revision, but must be updated alongside Resonate plugin
    # Find all ros BAG files on system
    data_file_names = []
    for folder in top_folders:
        top = folder
        for root, dirs, files in os.walk(top, topdown=False):
            for name in files:
                if name.endswith(("recording", ".bag")):
                    data_file_names.append(os.path.join(root, name))
    datafiles = []
    
    if recent_execution and (file_path is not None):
        df = df_class()
        df.read(file_path)
        datafiles.append(df)
    elif recent_execution:
        df = df_class()
        df.read(data_file_names[-1])
        datafiles.append(df)
    else:
        if (len(data_file_names) ==0):
            print('No data files found...Quitting...')
            return
        print("Starting to process the data files")
        for file_name in tqdm(data_file_names):
            df = df_class()
            df.read(file_name)
            datafiles.append(df)
        print("Finished processing the data files")
    print("\n")
    print("Computing Statistics .....")

    calculate_stats(datafiles)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="calculate stats for resonate\n", formatter_class=RawTextHelpFormatter)
    parser.add_argument("path", help="specify the location of the data to run analysis on", type=str)
    parser.add_argument("--datafile", "-df", default="uuv", help="Select the type of datafile to use")

    arguments = parser.parse_args()

    if arguments.datafile.lower() == "uuv":
        _data_file_names = glob.glob(os.path.join(arguments.path, "*.bag"))
        Datafile = UUVDatafile
    elif arguments.datafile.lower() == "carla":
        _data_file_names = glob.glob(os.path.join(arguments.path, "*.csv"))
        Datafile = CARLADatafile
    else:
        raise IOError("unrecognized datafile type")

    # TODO: This is faster with Python Multiprocessing, but serialization issues must be solved first.
    # See https://answers.ros.org/question/374301/slow-reading-from-many-rosbags-in-python-no-speedup-from-parallelization/
    _datafiles = []
    print("Starting to process the data files")
    for _file_path in tqdm(_data_file_names):
        _df = Datafile()
        _df.read(_file_path)
        _datafiles.append(_df)
    print("Finished processing the data files")
    print("\n")
    
    print(" Statistics .....")
    calculate_stats(_datafiles)
