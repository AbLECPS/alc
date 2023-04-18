# !/usr/bin/env python2
# Author: Charlie Hartsell <charles.a.hartsell@vanderbilt.edu>
"""This module defines the FileDownloader utility class for uploading files to a remote fileserver."""
from __future__ import print_function

import os
import sys
import paramiko
import time
import functools
import threading
import alc_utils.config as alc_config
import alc_utils.common as alc_common
import json


class FileUploader:
    """This class provides utility functions for uploading files to a specified fileserver and generating a
    corresponding storage metadata """

    def __init__(self):
        # Init class variables
        self.uploading_lock = threading.Lock()
        self.files_uploading = {}

    def upload_with_params(self, local_dir, param_dict):
        """Convenience wrapper around upload() function which reads upload options from a parameter dictionary"""
        # Check if parameters specify an upload at all. If not, hash local directory and return metadata.
        if not param_dict.get("upload", False):
            local_dir_hash = alc_common.hash_directory(local_dir)
            storage_metadata = {"upload_prefix": None,
                                "directory": local_dir,
                                "description": "No Upload",
                                "hash": local_dir_hash}
            return storage_metadata

        # Otherwise, read parameters from provided dictionary
        upload_desc = param_dict.get("description", None)
        sftp_config = param_dict.get(
            "sftp_config_file", alc_config.file_management_defaults.SFTP_CONFIG_FILE)
        preserve_dir_count = param_dict.get(
            "preserve_dir_count", alc_config.file_management_defaults.PRESERVE_DIR_COUNT)
        relative_base_dir = param_dict.get(
            "relative_base_dir", alc_config.file_management_defaults.STORAGE_BASE_PATH)
        path_prefix = param_dict.get("prefix", None)
        path_prefix = param_dict.get("path_prefix", path_prefix)
        path_prefix = param_dict.get("fs_path_prefix", path_prefix)
        path_prefix = param_dict.get("upload_path_prefix", path_prefix)

        # Perform upload and return storage metadata
        (error_code, storage_metadata) = self.upload(local_dir,
                                                     sftp_config_file=sftp_config,
                                                     description=upload_desc,
                                                     additional_prefix=path_prefix,
                                                     preserve_dir_count=preserve_dir_count,
                                                     relative_base_dir=relative_base_dir)
        return storage_metadata

    def upload(self, dir_full_path,
               sftp_config_file=alc_config.SFTP_CONFIG_FILE,
               description=None,
               additional_prefix=None,
               preserve_dir_count=alc_config.file_management_defaults.PRESERVE_DIR_COUNT,
               relative_base_dir=alc_config.file_management_defaults.STORAGE_BASE_PATH):
        """This function will upload the given directory to a remote fileserver and return a corresponding storage
        metadata.

        Details about the remote fileserver are specified in the SFTP config file."""
        # Read SFTP config file
        with open(sftp_config_file, 'r') as sftp_fp:
            sftp_config_dict = json.load(sftp_fp)

        # Strip any trailing path separator from directory path
        dir_full_path = alc_common.strip_trailing_separator(dir_full_path)

        dirname, basename = split_directory_path(dir_full_path,
                                                 preserve_dir_count=preserve_dir_count,
                                                 split_base_path=relative_base_dir)

        # Determine full upload path prefix
        if additional_prefix is not None:
            # Strip any leading path separator from additional prefix
            additional_prefix = os.path.normpath(additional_prefix)
            if additional_prefix.startswith(os.path.sep):
                additional_prefix = additional_prefix[1:]
            upload_prefix = os.path.join(
                sftp_config_dict.get("path_prefix"), additional_prefix)
        else:
            upload_prefix = sftp_config_dict.get("path_prefix")

        # Upload directory
        (result_code, upload_metadata) = self.upload_directory_to_fileserver(dirname,
                                                                             basename,
                                                                             upload_prefix,
                                                                             os.path.expandvars(
                                                                                 sftp_config_dict.get("key")),
                                                                             sftp_config_dict.get(
                                                                                 "host"),
                                                                             sftp_config_dict.get(
                                                                                 "port"),
                                                                             sftp_config_dict.get(
                                                                                 "user"),
                                                                             description)

        # Construct and return JSON metadata
        if result_code == 0:
            return result_code, upload_metadata
        else:
            return result_code, {}

    def upload_directory_to_fileserver(self, local_dirname,
                                       local_basename,
                                       upload_dir_prefix,
                                       key_path,
                                       host,
                                       port,
                                       user,
                                       description):
        # DEBUG
        # paramiko.common.logging.basicConfig(level=paramiko.common.DEBUG)

        # Read private key file
        temp_str = str.format("Reading private key file ({})", key_path)
        print(temp_str)
        private_key = paramiko.RSAKey(filename=key_path)

        # Establish SSH connection
        temp_str = str.format(
            "Connecting to host ({}, {}) as user {}", host, port, user)
        print(temp_str)
        ssh_transport = paramiko.Transport((host, port))
        ssh_transport.connect(username=user, pkey=private_key)

        # Get SFTP Client over SSH connection
        sftp_client = paramiko.SFTPClient.from_transport(ssh_transport)

        # Normalize paths and trim any trailing path separators
        local_dirname = alc_common.strip_trailing_separator(local_dirname)
        local_basename = alc_common.strip_trailing_separator(local_basename)
        upload_dir_prefix = alc_common.strip_trailing_separator(
            upload_dir_prefix)

        # Hash local directory. Hash digest is saved in storage metadata
        local_dir_fullname = os.path.join(local_dirname, local_basename)
        local_dir_hash = alc_common.hash_directory(local_dir_fullname)

        # Make sure desired upload directory exists on fileserver
        upload_dir_fullname = os.path.join(upload_dir_prefix, local_basename)
        upload_dir_prefix_list = split_path(upload_dir_fullname)
        temp_dir = ""
        for next_dir in upload_dir_prefix_list:
            temp_dir = os.path.join(temp_dir, next_dir)
            try:
                sftp_client.mkdir(temp_dir)
            except IOError as e:
                if e.errno is None:
                    # This typically occurs when Directory already exists on fileserver
                    # Probably need a better solution for this.
                    print("IOError with no errno returned when creating directory " + upload_dir_prefix +
                          " on fileserver. This usually means directory already exists.")
                else:
                    raise e

        # Copy local directory to server
        # Get absolute path of results directory and determine prefix (Everything before this run's results directory)
        error_during_upload = False
        dir_prefix_len = len(local_dirname) + 1
        for root, dirs, files in os.walk(local_dir_fullname):
            # Determine 'basename' of current directory. (Path relative to local_dirname)
            basename = root[dir_prefix_len:]

            # Determine remote directory and filename and ensure directory exists on fileserver
            remote_directory_name = os.path.join(upload_dir_prefix, basename)
            try:
                sftp_client.mkdir(remote_directory_name)
            except IOError as e:
                if e.errno is None:
                    # This typically occurs when Directory already exists on fileserver
                    # Probably need a better solution for this.
                    print("IOError with no errno returned when creating directory " + remote_directory_name +
                          " on fileserver. This usually means directory already exists.")
                else:
                    raise e

            # Copy each file in this directory to fileshare
            for f in files:
                short_filename = f
                local_filename = os.path.join(root, f)
                remote_filename = os.path.join(remote_directory_name, f)

                # Copy local file to fileserver
                try:
                    bound_put_cb = functools.partial(
                        self.put_cb, short_filename)
                    sftp_client.put(
                        local_filename, remote_filename, callback=bound_put_cb)
                except IOError as e:
                    # FIXME: Should we abort if we get an IOError instead of ignoring?
                    print("IOError when copying to fileserver")
                    error_during_upload = True

        # Simple poller loop to periodically check if uploads have finished
        # FIXME: Add Timeout -- does paramiko library allow for cancelling uploads in progress?
        uploads_in_progress = True
        while uploads_in_progress:
            time.sleep(1)

            uploads_in_progress = False
            self.uploading_lock.acquire()
            for key in self.files_uploading:
                if self.files_uploading[key] < 1:
                    uploads_in_progress = True
                    break
            self.uploading_lock.release()

        # Construct upload metadata
        upload_metadata = {"upload_prefix": upload_dir_prefix,
                           "directory": local_basename,
                           "description": description,
                           "hash": local_dir_hash}

        if error_during_upload:
            print("************************* WARNING ****************************")
            print(
                "WARNING: File uploads completed with errors. Uploaded data may be corrupted.")
        else:
            print("File uploads complete.")

        # Close connections and return uploaded directory name
        sftp_client.close()
        ssh_transport.close()
        return 0, upload_metadata

    def put_cb(self, filename, bytes_transferred, total_bytes):
        # Lock ensures only one instance of callback access data struct at a time
        self.uploading_lock.acquire()

        # Check if this is first callback update for this filename
        next_update_percentage = self.files_uploading.get(filename)
        if next_update_percentage is None:
            next_update_percentage = .10
            self.files_uploading[filename] = next_update_percentage
            sys.stdout.write(str.format(
                "Uploading {0} [{1} bytes]: ", filename, str(total_bytes)))
            sys.stdout.flush()

        # Occasionally, a file may not contain anything. Not sure why, but check to prevent division by 0.
        if total_bytes > 0:
            current_percentage = bytes_transferred / float(total_bytes)
        else:
            current_percentage = 1

        # Sometimes callback is called twice on completion. Make sure we only finish once.
        if next_update_percentage >= 1:
            pass
        # Check if upload is complete (first time)
        elif bytes_transferred >= total_bytes:
            print("DONE")
            self.files_uploading[filename] = 1
        # Check if we should print(upload progress)
        elif current_percentage >= next_update_percentage:
            # print(percentage, rounded to nearest 10%)
            rounded_percentage = round(current_percentage * 10)
            rounded_percentage *= 10
            sys.stdout.write(str.format(
                "...{0}% ", str(int(rounded_percentage))))
            sys.stdout.flush()

            # Update user on upload progress every 10%
            next_update_percentage = (rounded_percentage / 100) + .10
            self.files_uploading[filename] = next_update_percentage

        # Callback is done, unlock mutex
        self.uploading_lock.release()


def split_path(p):
    a, b = os.path.split(p)
    if (len(a) > 0) and (len(b) > 0):
        retval = split_path(a) + [b]
    else:
        retval = [b]
    return retval


def split_directory_path(dir_full_path,
                         split_base_path=alc_config.file_management_defaults.STORAGE_BASE_PATH,
                         preserve_dir_count=alc_config.file_management_defaults.PRESERVE_DIR_COUNT):
    """Split the provided directory path into two parts: dir_name and base_name. If preserve_dir_count is None,
    split directory such that dir_name = split_base_path and base_name is the remainder of the path. """
    # Convert both paths to real paths to eliminate any possible issues with symbolic links.
    # Remove trailing separator from split_base_path for consistency
    dir_full_path = os.path.realpath(dir_full_path)
    split_base_path = os.path.realpath(split_base_path)
    split_base_path = alc_common.strip_trailing_separator(split_base_path)

    if preserve_dir_count is None:
        if dir_full_path.startswith(split_base_path):
            # Trailing slash of split_base_path should be part of "dirname". "basename" should NOT have a leading slash.
            dirname = split_base_path + os.path.sep
            basename_start_idx = len(split_base_path) + 1
            basename = dir_full_path[basename_start_idx:]
        else:
            raise IOError("Desired directory to upload (%s) does not appear to be contained within the provided base"
                          "path (%s). Please check that provided paths are correct.", dir_full_path, split_base_path)

    # If preserve_dir_count is not None, split path by keeping <preserve_dir_count> layers of the path hierarchy as
    # part of the "base_path". Remainder of path is "dir_path"
    else:
        if preserve_dir_count < 1:
            raise ValueError(
                "preserve_dir_count less than 1, but FileUploader must preserve at least 1 level of directory naming")

        # Determine name of directory to be uploaded, taking into account desired number of levels of the directory
        # hierarchy to preserve eg. "/my/example/path/to/upload" with preserve_dir_count=2 should yield "to/upload"
        dirname = dir_full_path
        basename = ""
        for i in range(0, preserve_dir_count):
            (dirname, tail) = os.path.split(dirname)
            if (dirname is None) and (tail is None):
                raise ValueError("preserve_dir_count (%d) exceeded maximum value (%d) for path (%s)" %
                                 (preserve_dir_count, i, dir_full_path))
            basename = os.path.join(tail, basename)

    return dirname, basename
