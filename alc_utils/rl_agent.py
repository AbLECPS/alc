# !/usr/bin/env python2
# Author: Charlie Hartsell <charles.a.hartsell@vanderbilt.edu>
'''This module defines the RLAgentBase base class for defining a reinforcement-learning agent.'''

import imp
import os
import sys

DEFAULT_RL_AGENT_MODULE_NAME = "RLAgent"


# Function load an implementation of RLAgent, which should be a subclass of RLAgentBase
def load(module_dir, module_name=DEFAULT_RL_AGENT_MODULE_NAME):
    '''Loads an implementation of RLAgent from the specified module directory

    The RLAgent being loaded should be a derived class of RLAgentBase
    '''
    rl_agent_module_path = os.path.join(module_dir, module_name + '.py')

    # Input checks
    assert os.path.isfile(rl_agent_module_path), 'RLAgent module file {} is not a valid file'.format(
        rl_agent_module_path)

    # Add module directory to sys.path and load
    sys.path.append(os.path.abspath(module_dir))
    rl_agent_module = imp.load_source(module_name, rl_agent_module_path)

    # Make sure this is a subclass of RLAgentBase
    # Raises TypeError exception otherwise
    issubclass(rl_agent_module.RLAgent, RLAgentBase)

    # Create and return  the RLAgent class
    return rl_agent_module.RLAgent


class RLAgentBase:
    '''Base class for RLAgent. Functions are abstract and must be implemented by a derived class.'''

    def __init__(self):
        raise NotImplementedError(
            "Abstract __init__() function of RLAgentBase class called.")

    def save(self):
        '''Saves the current state of the RLAgent'''
        raise NotImplementedError(
            "Abstract save() function of RLAgentBase class called.")

    def step(self, observation, reward, done):
        '''Determines an action based on the current observations and reward value'''
        raise NotImplementedError(
            "Abstract step() function of RLAgentBase class called.")
