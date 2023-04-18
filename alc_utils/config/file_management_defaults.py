# !/usr/bin/env python
# Author: Charlie Hartsell <charles.a.hartsell@vanderbilt.edu>
"""Default parameter configuration for ALC Utilities file management utilities"""
from . import base_config

UPLOAD_TIMEOUT = 120  # Allow maximum of 2 minutes for uploading
SFTP_CONFIG_FILE = base_config.SFTP_CONFIG_FILE
PRESERVE_DIR_COUNT = None
STORAGE_BASE_PATH = base_config.WORKING_DIRECTORY
