# !/usr/bin/env python2
# Author: Charlie Hartsell <charles.a.hartsell@vanderbilt.edu>
'''This module defines the RLEnvironmentBase base class for defining a reinforcement-learning environment.'''

import imp
import os
import sys

DEFAULT_RL_ENVIRONMENT_MODULE_NAME = "RLEnvironment"


# Function load an implementation of RLEnvironment, which should be a subclass of RLEnvironmentBase
def load(module_dir, module_name=DEFAULT_RL_ENVIRONMENT_MODULE_NAME):
    '''Loads an implementation of RLEnvironment from the specified module directory

    The RLEnvironment being loaded should be a derived class of RLEnvironmentBase
    '''
    rl_env_module_path = os.path.join(module_dir, module_name + '.py')

    # Input checks
    assert os.path.isfile(
        rl_env_module_path), 'RLEnvironment module file {} is not a valid file'.format(rl_env_module_path)

    # add to sys.path
    sys.path.append(os.path.abspath(module_dir))

    rl_env_module = imp.load_source(module_name, rl_env_module_path)

    # Make sure this is a subclass of RLEnvironmentBase
    # Raises TypeError exception otherwise
    issubclass(rl_env_module.RLEnvironment, RLEnvironmentBase)

    # Create and return the RLEnvironment class
    return rl_env_module.RLEnvironment


class RLEnvironmentBase:
    '''Base class for RLEnvironment. Functions are abstract and must be implemented by a derived class.'''

    def __init__(self):
        raise NotImplementedError(
            "Abstract __init__() function of RLEnvironmentBase class called.")

    def done(self):
        '''Returns true if the learning episode is complete'''
        raise NotImplementedError(
            "Abstract done() function of RLEnvironmentBase class called.")

    def reward(self, observation, reward, done):
        '''Calculates a reward value based on the current state'''
        raise NotImplementedError(
            "Abstract reward() function of RLEnvironmentBase class called.")
