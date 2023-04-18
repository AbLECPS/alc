#!/usr/bin/env python
import os
import rospy
import numpy as np
import csv

class CSVLogger(object):
    '''
    CSV Data Logger
    '''
    def __init__(self, filename, results_dir):
        self.log_filename = rospy.get_param('~log_filename', filename)
        self.results_dir = results_dir

    def write_to_file(self, data, new_file=False):
        if os.path.isdir(self.results_dir):
            if not new_file:
                mode = 'a'
            else:
                mode = 'w'
            with open(os.path.join(self.results_dir, self.log_filename), mode) as fd:
                writer = csv.writer(fd)
                writer.writerow(data)
                fd.close()
        else:
            print(self.results_dir)
            rospy.logwarn("[CSV_LOGGER] log file path error")
