# !/usr/bin/env python2
# Author: Charlie Hartsell <charles.a.hartsell@vanderbilt.edu>
"""Configuration file for ALC Utilities"""
import os
from . import config_util

# Get required environment variables
ALC_HOME = os.getenv('ALC_HOME')
if ALC_HOME is None:
    raise EnvironmentError("ALC_HOME environment variable is not defined.")
WORKING_DIRECTORY = os.getenv('ALC_WORKING_DIR')
if WORKING_DIRECTORY is None:
    raise EnvironmentError(
        "ALC_WORKING_DIR environment variable is not defined.")

# Get optional environment variables (or use default)
SFTP_CONFIG_FILE = os.getenv('ALC_FILESHARE_CONFIG_FILE', os.path.join(
    ALC_HOME, "alc_utils/config/sftp_default_config.json"))
ALC_TEST_DIR = os.getenv(
    "ALC_TEST_DIR", os.path.join(ALC_HOME, "alc_utils/test"))
# Determine any useful sub-directories
TEST_RESOURCE_DIR = os.path.join(ALC_TEST_DIR, "res")
DOWNLOAD_DIRECTORY = os.path.join(WORKING_DIRECTORY, 'downloads')
JUPYTER_WORK_DIR = os.path.join(WORKING_DIRECTORY, 'jupyter')
JUPYTER_MATLAB_WORK_DIR = os.path.join(WORKING_DIRECTORY, 'jupyter_matlab')
WEBGME_WORK_DIR = os.path.join(WORKING_DIRECTORY, 'webgme')
VERIFICATION_UTILS_DIR = os.path.join(ALC_HOME, 'verification', 'utils')
WORKSPACE_CACHE_DIR = os.path.join(WORKING_DIRECTORY, 'cache')

# Locations to search when checking if datasets are stored locally (before attempting to download from a fileserver)
_SEARCH_DIRS = [WORKING_DIRECTORY, DOWNLOAD_DIRECTORY, TEST_RESOURCE_DIR]
# Prune any 'None' entries
DOWNLOAD_SEARCH_DIRS = []
for _dir in _SEARCH_DIRS:
    if _dir is not None:
        DOWNLOAD_SEARCH_DIRS.append(_dir)

# Location of Machine Learning library (eg. Keras, PyTorch, etc.) adapter classes and default adapter path
ML_LIBRARY_ADAPTERS_DIR = os.path.join(
    ALC_HOME, "alc_utils/ml_library_adapters")
ML_LIBRARY_ADAPTER_PATH = os.path.join(
    ML_LIBRARY_ADAPTERS_DIR, "keras_library_adapter.py")

# Variables defined in this file should also be defined in the 'env' dictionary
# This is useful for interoperability with the OS-level environment variables
env = config_util.build_var_dict(globals())
