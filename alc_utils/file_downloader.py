# !/usr/bin/env python2
# Author: Charlie Hartsell <charles.a.hartsell@vanderbilt.edu>
"""This module defines the FileDownloader utility class for downloading files from a remote fileserver."""
from __future__ import print_function

import os
import sys
import paramiko
import time
import functools
import threading
import errno
import stat
import warnings
import json
import config as alc_config
import common as alc_common
from six import string_types


# FIXME: Should raise exceptions instead of returning codes
class FileDownloader:
    """This class provides utility functions for finding and downloading files based on provided storage metadata."""

    def __init__(self):
        self.files_downloading = {}
        self.downloading_lock = threading.Lock()

    # FIXME: Remove "local_data_directory" parameter. Make sure it is not used anywhere
    def download(self,
                 object_uri_list,
                 download_directory=alc_config.DOWNLOAD_DIRECTORY,
                 key_file=None,
                 sftp_config_file=alc_config.SFTP_CONFIG_FILE,
                 local_data_directory=None,
                 force_download=False):
        """This function will find/download a directory based on the provided storage metadata.

        If a directory matching the provided metadata is found locally, then no download is performed.
        Otherwise, the directory will be downloaded from the fileserver specified by the provided SFTP config file."""
        # Check if list of URIs or single URI is specified (URI == storage metadata)
        if isinstance(object_uri_list, dict):
            object_uri_list = [object_uri_list]
        elif isinstance(object_uri_list, list):
            pass
        else:
            # Unexpected type
            return -1, None

        # Read SFTP config file
        try:
            with open(sftp_config_file, 'r') as sftp_fp:
                sftp_config = json.load(sftp_fp)
        except IOError as e:
            # Only want to catch 'No such file or directory'. Re-raise otherwise.
            if e.errno != errno.ENOENT:
                raise e
            print("Could not find SFTP config file at %s. Continuing without SFTP."
                  % sftp_config_file)
            sftp_config = None

        # Find SSH key file if none specified
        if (key_file is None) and (sftp_config is not None):
            key_file = sftp_config["key"]

        # Loop over all URLs to be downloaded
        downloaded_dirs = []
        for object_uri in object_uri_list:
            # Input checks
            local_dir_name = object_uri["directory"]
            # 'basestring' encompasses 'str' and 'unicode' type strings
            if not (isinstance(local_dir_name, string_types)):
                raise TypeError(
                    "Provided download metadata contained non-string type directory entry")

            # FIXME: This is a workaround. Remove when no longer needed
            # For compatibility with URI's from older API versions, if an absolute path is provided
            # try to strip path until "jupyter" or "jupyter_matlab"
            if os.path.isabs(local_dir_name):
                start_idx = local_dir_name.find("jupyter")
                if start_idx > 0:
                    local_dir_name = local_dir_name[start_idx:]
            object_uri["directory"] = local_dir_name

            # Normalize and strip any trailing path separators
            local_dir_name = os.path.normpath(local_dir_name)
            if local_dir_name.endswith(os.path.sep):
                local_dir_name = local_dir_name[:-1]

            # Check if this metadata contains a "upload_prefix" field and determine remote directory name
            upload_prefix = object_uri.get("upload_prefix", -1)
            if upload_prefix == -1:
                # For storage metadata files without an upload_prefix,
                # assume local directory name is basename of remote dir
                remote_dir = local_dir_name
                local_dir_name = os.path.basename(remote_dir)
                warnings.warn(
                    "Storage metadata without 'upload_prefix' field is deprecated.", DeprecationWarning)
            elif upload_prefix is None:
                # For storage metadata files for local directory that is not uploaded
                remote_dir = local_dir_name
                local_dir_name = remote_dir
            else:
                if local_dir_name.startswith(os.path.sep):
                    local_dir_name = local_dir_name[1:]
                remote_dir = os.path.join(upload_prefix, local_dir_name)

            # Check if data is already stored locally in any search path (unless force_download flag is set)
            if not force_download:
                print("Searching for directory '%s' locally..." %
                      local_dir_name)
                # FIXME: Need to be able to set ignore_hash_check as argument
                #       Should also default to False, not True (ie. perform hash check by default)
                local_dir_match = alc_common.search_for_uri_locally(
                    object_uri, ignore_hash_check=True)
                if local_dir_match is not None:
                    # If data is found locally, append to downloaded directories and skip downloading step
                    print("Found directory '%s' locally." % local_dir_name)
                    downloaded_dirs.append(local_dir_match)
                    continue
                else:
                    print("Directory '%s' not found locally." % local_dir_name)

            # TODO: Test this
            # Data needs to be re-downloaded. Make sure any existing local files in the download location are removed.
            # local_dir = os.path.join(download_directory, remote_dir)
            # try:
            #     for file_name in os.listdir(local_dir):
            #         file_path = os.path.join(local_dir, file_name)
            #         if os.path.isfile(file_path):
            #             os.unlink(file_path)
            #         elif os.path.isdir(file_path): shutil.rmtree(file_path)
            # # FIXME: Make this exception less broad
            # except Exception as e:
            #     # TODO: catch path not found exception
            #     print e

            # Download directory from fileserver and add to list of downloaded directories
            if sftp_config is not None:
                result_code, downloaded_dir = self.download_directory_from_fileserver(download_directory,
                                                                                      remote_dir,
                                                                                      key_file,
                                                                                      object_uri.get(
                                                                                          "hash", None),
                                                                                      sftp_config)
            else:
                result_code = 1

            if result_code != 0:
                return result_code, downloaded_dirs
            else:
                downloaded_dirs.append(downloaded_dir)

        return 0, downloaded_dirs

    def download_directory_from_fileserver(self, local_dir, remote_dir, key_path, expected_hash, sftp_config):
        # Get SFTP server info
        host = sftp_config["host"]
        port = sftp_config["port"]
        user = sftp_config["user"]

        # Read private key file
        print("Reading private key file (%s)." % key_path)
        try:
            private_key = paramiko.RSAKey(filename=key_path)
        except IOError as e:
            print("Failed reading private key file (%s)", key_path)
            return (-1, None)
        except paramiko.PasswordRequiredException as e:
            print("Private key file (%s) requires a password", key_path)
            return (-1, None)
        except Exception as e:
            print("Failed reading private key file (%s)", key_path)
            print("Exception: %s", e.__str__())
            return (-1, None)

        # Establish SSH connection
        tempStr = str.format("Connecting to host ({}, {})", host, port)
        print(tempStr)
        ssh_transport = paramiko.Transport((host, port))
        try:
            ssh_transport.connect(username=user, pkey=private_key)
        except paramiko.SSHException as e:
            print("SSH authentication to host (%s, %s) failed.", host, port)
            return (-1, None)
        except Exception as e:
            print("Failed to establish SSH connection with host (%s, %s)", host, port)
            print("Exception: %s", e.__str__())
            return (-1, None)

        # Get SFTP Client over SSH connection
        try:
            sftp_client = paramiko.SFTPClient.from_transport(ssh_transport)
        except Exception as e:
            tempStr = str.format(
                "Failed to establish SFTP connection with host ({}, {})", host, port)
            print(tempStr)
            print("Exception: ", e.__str__())
            return (-1, None)

        # Make sure local directory exists so we have a place to save the downloaded data
        try:
            os.makedirs(local_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise e

        # Search specified remote directory and all sub-directories to construct list of filenames (absolute) to be downloaded
        if remote_dir.startswith('/') or remote_dir.startswith('\\'):
            remote_dir = remote_dir[1:]
        remote_directories = [remote_dir]
        remote_filenames = []
        while len(remote_directories) > 0:
            # Read contents of next directory
            next_dir = remote_directories.pop(0)
            try:
                dir_contents = sftp_client.listdir_attr(next_dir)
            except IOError as e:
                print("IOError when reading file attributes of directory %s from fileserver: %s" % (
                    next_dir, str(e)))
                return (-1, None)

            # For each file: if it is a directory, then add it to directories list. Otherwise, add file to download list
            for fileattr in dir_contents:
                full_filename = os.path.join(next_dir, fileattr.filename)
                if stat.S_ISDIR(fileattr.st_mode):
                    remote_directories.append(full_filename)
                else:
                    remote_filenames.append(full_filename)

        # Manipulate filenames/paths as needed and download file
        if remote_dir.endswith('/'):
            remote_dir = remote_dir[:-1]
        remote_dir_prefix = os.path.split(remote_dir)[0]
        remote_dir_basename = os.path.split(remote_dir)[1]
        remote_dir_prefix_len = len(remote_dir_prefix)
        for full_remote_filename in remote_filenames:
            # Determine relative file name (relative to specified remote directory argument)
            relative_remote_filename = full_remote_filename[remote_dir_prefix_len:]
            short_filename = os.path.split(relative_remote_filename)[1]

            # Determine appropriate local directory and filename, and ensure directory exists
            full_local_filename = os.path.join(local_dir, full_remote_filename)
            full_local_dir = os.path.split(full_local_filename)[0]
            try:
                os.makedirs(full_local_dir)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

            # Download file from fileserver
            try:
                bound_get_cb = functools.partial(self.get_cb, short_filename)
                sftp_client.get(remotepath=str(full_remote_filename), localpath=str(full_local_filename),
                                callback=bound_get_cb)
            except IOError as e:
                print("IOError when downloading file from fileserver: " + str(e))
                return (-1, None)

        # Simple poller loop to periodically check if all uploads have finished
        downloads_in_progress = True
        while downloads_in_progress:
            time.sleep(1)

            downloads_in_progress = False
            self.downloading_lock.acquire()
            for key in self.files_downloading:
                if self.files_downloading[key] < 1:
                    downloads_in_progress = True
                    break
            self.downloading_lock.release()
        print("File downloads complete.")

        # Verify that hash of downloaded directory matches expected value
        downloaded_dir = os.path.join(local_dir, remote_dir)
        if expected_hash is not None:
            downloaded_hash = alc_common.hash_directory(downloaded_dir)
            if expected_hash == downloaded_hash:
                print("Hash of downloaded files matches expected value.")
            else:
                print("WARNING: Hash of downloaded files does NOT match expected value.")

        # Close connections and return
        sftp_client.close()
        ssh_transport.close()
        return 0, os.path.join(local_dir, remote_dir)

    def get_cb(self, filename, bytesTransferred, totalBytes):
        # Lock ensures only one instance of callback access data struct at a time
        self.downloading_lock.acquire()

        # Check if this is first callback update for this filename
        next_update_percentage = self.files_downloading.get(filename)
        if next_update_percentage is None:
            next_update_percentage = .10
            self.files_downloading[filename] = next_update_percentage
            tempStr = str.format(
                "Downloading {0} [{1} bytes]: ", filename, str(totalBytes))
            sys.stdout.write(tempStr)
            sys.stdout.flush()

        # Occasionally, a file may not contain anything. Not sure why, but check to prevent division by 0.
        if totalBytes > 0:
            current_percentage = bytesTransferred / float(totalBytes)
        else:
            current_percentage = 1

        # Sometimes callback is called twice on completion. Make sure we only finish once.
        if next_update_percentage >= 1:
            pass
        # Check if upload is complete (first time)
        elif bytesTransferred >= totalBytes:
            print("DONE")
            self.files_downloading[filename] = 1
        # Check if we should print upload progress
        elif current_percentage >= next_update_percentage:
            # Print percentage, rounded to nearest 10%
            rounded_percentage = round(current_percentage * 10)
            rounded_percentage *= 10
            tempStr = str.format("...{0}% ", str(int(rounded_percentage)))
            sys.stdout.write(tempStr)
            sys.stdout.flush()

            # Update user on upload progress every 10%
            next_update_percentage = (rounded_percentage / 100) + .10
            self.files_downloading[filename] = next_update_percentage

        # Callback is done, unlock mutex
        self.downloading_lock.release()
