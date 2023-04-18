#!/usr/bin/env python
# Author: Charlie Hartsell <charles.a.hartsell@vanderbilt.edu>
"""This file defines a convenience class for loading and running multiple AssuranceMonitors."""
from assurance_monitor import AssuranceMonitor
import os
import glob


ASSURANCE_MONITOR_NAME = "MultiAssuranceMonitor"


class MultiAssuranceMonitor:
    def __init__(self):
        self.assurance_monitors = None
        self.ood_detectors = None
        
    def load(self, directory_list):
        """Loads and stores one assurance monitor from each directory in directory_list."""
        # Clear any assurance monitors loaded previously
        self.assurance_monitors = []

        # Load assurance monitor from each directory
        for directory in directory_list:
            # Search each directory for multiple assurance monitor folders
            sub_dirs = []
            candidates = glob.glob(os.path.join(directory, "assurance_monitor*"))
            candidates.sort(key=os.path.getmtime, reverse=True)
            print(candidates)
            for candidate in candidates:
                if os.path.isdir(candidate):
                    sub_dirs.append(candidate)
                    print(candidate)
                    break

            # Require that every directory have at least one valid candidate subdirectory
            if len(sub_dirs):
                self.assurance_monitors.append(AssuranceMonitor.load(sub_dirs[0]))
            else:
                self.assurance_monitors.append(AssuranceMonitor.load(directory))
        
        self.ood_detectors = self.assurance_monitors

    def evaluate(self, input_data, *args, **kwargs):
        """Call evaluate function for each loaded AssuranceMonitor and return first valid result,
        or 'None' if no results are valid"""
        for monitor in self.assurance_monitors:
            result = monitor.evaluate(input_data, *args, **kwargs)
            if result is not None:
                return result

        return None
