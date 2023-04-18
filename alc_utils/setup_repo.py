import sys
import traceback
import logging
import json
import re
import time
import os
import git
from git import Repo
from pathlib import Path
import subprocess
from subprocess import Popen, PIPE
import shutil
import tarfile


alc_working_dir_env_var_name = "ALC_WORKING_DIR"
docker_dir_name = "docker"
user_dir_name = "users"
docker_prefix = "codeserver"
git_server_url = '172.23.0.1'
git_server_port = '2222'
git_server_path = 'git-server/repos/'


class RepoSetup:

    def __init__(self):
        self.git_server_url = os.environ.get(
            "ALC_GITSERVER_HOST", git_server_url)
        self.git_server_port = os.environ.get(
            "ALC_GITSERVER_PORT", git_server_port)

    def clone_repo(self, parent_folder, repo_info, branch_info, tag_info, logger, force=False, ignore_errors=False):
        try:
            p = parent_folder
            dst_folder_path = os.path.join(str(parent_folder), repo_info)
            if (os.path.exists(dst_folder_path) and force):
                shutil.rmtree(dst_folder_path)

            #logger.info(' destination folder for clone {0}'.format(dst_folder_path))
            print(' destination folder for clone {0}'.format(dst_folder_path))
            if (not p.exists()):
                p.mkdir(parents=True)
            git_server_str = 'ssh://git@{0}:{1}/{2}{3}.git'.format(
                self.git_server_url, self.git_server_port, git_server_path, repo_info)

            if (not os.path.exists(dst_folder_path)):
                repo = Repo.clone_from(git_server_str, dst_folder_path)
                o = repo.remotes.origin
                o.pull()
                if (not branch_info and tag_info):
                    branch_info = tag_info
                if (branch_info):
                    repo.git.checkout(branch_info)
                #o = repo.remotes.origin
                # o.pull()

            else:
                repo = Repo(dst_folder_path)
                repo.git.checkout('master')
                o = repo.remotes.origin
                o.pull()
                if (not branch_info and tag_info):
                    branch_info = tag_info
                if (branch_info):
                    repo.git.checkout(branch_info)
                #o = repo.remotes.origin
                # o.pull()

        except Exception as err:
            #logger.error(' Error encountered while setting up the repo at {0}'.format(dst_folder_path))
            msg = str(err)
            print(msg)
            #logger.info("exception {0}".format(msg))
            traceback_msg = traceback.format_exc()
            print(traceback_msg)
            # logger.info(traceback_msg)
            sys_exec_info_msg = sys.exc_info()[2]
            print(sys_exec_info_msg)
            # logger.info(sys_exec_info_msg)
            if (ignore_errors):
                return False
            raise
        return True

    def archive_repo(self, tar_file_name, parent_folder, repo_info, branch_info, tag_info, logger, ignore_errors=False):
        try:
            if (self.clone_repo(parent_folder, repo_info, branch_info, tag_info, logger, ignore_errors)):
                dst_folder_path = os.path.join(str(parent_folder), repo_info)
            topreserve = ['.git', 'alc_utils', 'alc_ros', 'activities']

            for i in topreserve:
                if os.path.exists(os.path.join(dst_folder_path, i)):
                    print('removing '+i)
                    shutil.rmtree(os.path.join(dst_folder_path, i))

            #repo = Repo(dst_folder_path)
            # with open(tar_file_name, 'wb') as fp:
            #    repo.archive(fp)
            with tarfile.open(tar_file_name, "w:gz") as tar:
                tar.add(dst_folder_path, arcname='')
            shutil.rmtree(dst_folder_path)

        except Exception as err:
            #logger.error(' Error encountered while archiving up the repo at {0}'.format(dst_folder_path))

            msg = str(err)
            print(msg)
            #logger.info("exception {0}".format(msg))
            traceback_msg = traceback.format_exc()
            print(traceback_msg)
            # logger.info(traceback_msg)
            sys_exec_info_msg = sys.exc_info()[2]
            print(sys_exec_info_msg)
            # logger.info(sys_exec_info_msg)
            if (ignore_errors):
                return False
            raise
        return True

    def build_repo(self, parent_folder, repo_info, logger):
        try:
            dst_folder_path = os.path.join(parent_folder, repo_info)
            if (not os.path.exists(dst_folder_path)):
                raise Exception(
                    "Build Error: Path does not exist {0} ".format(dst_folder_path))

            setup_script = os.path.join(dst_folder_path, 'setup.sh')
            build_script = os.path.join(dst_folder_path, 'build.sh')

            if (not os.path.exists(setup_script)):
                setup_script = build_script
            if (not os.path.exists(setup_script)):
                raise Exception(
                    "Build Error: setup.sh or build.sh was not found in the repo top folder {0} ".format(dst_folder_path))

            #session = subprocess.Popen([setup_script], stdout=PIPE, stderr=PIPE,cwd=dst_folder_path)
            #stdout, stderr = session.communicate()
            curr = os.getcwd()
            os.chdir(dst_folder_path)
            if (logger):
                logger.info(' script to be executed -'+setup_script)
            else:
                print(' script to be executed -'+setup_script)

            os.system(setup_script)
            os.chdir(curr)
            if (logger):
                logger.info(' done executing')
            else:
                print(' done executing')

            #logger.info('Build Output -------------')

            # if (stdout):
            #    logger.info('stdout')
            #    with open(os.path.join(dst_folder_path,'stdout.txt'), 'w') as file:
            #        file.write(str(stdout))

            #logger.info ('Build Error -------------')
            # if (stderr):
            #    logger.info('stderr')
            #    with open(os.path.join(dst_folder_path,'stderr.txt'), 'w') as file:
            #        file.write(str(stderr))

            # if stderr:
            #    raise Exception("Build Error "+str(stderr))

            # return stdout, stderr
            return "", ""

        except Exception as err:
            if (logger):
                logger.error(
                    ' Error encountered while building the repo at {0}'.format(dst_folder_path))
            else:
                print(' Error encountered while building the repo at {0}'.format(
                    dst_folder_path))
            raise

    def add_to_repo(self, parent_folder, msg, user, logger):
        try:
            if parent_folder.exists():
                dst_folder_path = str(parent_folder)
                repo = Repo(dst_folder_path)
                repo.config_writer().set_value("user", "name", user).release()
                repo.config_writer().set_value(
                    "user", "email", "{0}@alc.alc".format(user)).release()
                repo.git.add('-A')
                repo.index.commit(msg)
                repo.remotes.origin.push()
        except Exception as err:
            print(' Error encountered while commiting to the repo {0}'.format(
                str(parent_folder)))
            #logger.error(' Error encountered while commiting to the repo {0}'.format(str(parent_folder)))
            raise

    def create_branch_tag(self,
                          parent_folder,
                          repo_info,
                          src_branch,
                          src_tag,
                          new_branch,
                          new_tag,
                          logger,
                          ignore_errors=False):
        setup_status = self.clone_repo(
            self, parent_folder, repo_info, src_branch, src_tag, logger, ignore_errors=True)
        if (not setup_status):
            return setup_status, "repo setup failed"
        dst_folder_path = os.path.join(str(parent_folder), repo_info)
        try:
            repo = Repo(dst_folder_path)
            if (new_branch):
                logger.info('trying to create new branch {0} from branch {1}'.format(
                    new_branch, src_branch))
                current = repo.create_head(new_branch)
                current.checkout()
                repo.git.add(A=True)
                repo.git.commit(m='created new branch {0}  from branch {1} '.format(
                    new_branch, src_branch_info))
                repo.git.push('--set-upstream', 'origin', current)
                logger.info('finished creatingnew branch {0} from branch {1}'.format(
                    new_branch, src_branch))
                return True
            elif (new_tag):
                logger.info('trying to create new tag {0} '.format(new_tag))
                new_tag_info = repo.create_tag(
                    new_tag, message='tag "{0}" created'.format(new_tag))
                repo.remotes.origin.push(new_tag_info)
                logger.info('finished creating new tag {0} '.format(new_tag))
                return True

        except Exception as err:
            logger.error(
                ' Error encountered while setting up branch / tag for the repo')
            msg = str(err)
            logger.info("exception {0}".format(msg))
            traceback_msg = traceback.format_exc()
            logger.info(traceback_msg)
            sys_exec_info_msg = sys.exc_info()[2]
            logger.info(sys_exec_info_msg)
            return False, "Error encountered while creating branch/ tag"

        return False, "unknown issues"

    def create_new_repo(self, new_repo_info, contents_folder, logger, force=False, ignore_errors=False):

        msg = ''
        try:
            alc_working_dir_name = os.environ.get(
                alc_working_dir_env_var_name, None)
            if not alc_working_dir_name:
                msg = ' create_new_repo -> environment variable {0} is not defined'.format(
                    alc_working_dir_env_var_name)
                # logger.error(msg)
                print(msg)
                return False, msg

            repo_dir = Path(alc_working_dir_name, '.' +
                            git_server_path, new_repo_info+".git")

            if (repo_dir.exists() and not force):
                msg = ' repo  {0} already exists'.format(new_repo_info)
                # logger.error(msg)
                print(msg)
                return False, msg

            force_repo = 1
            cmd = '$ALC_HOME/alc_utils/docker/create_repo.sh {0}   {1}   {2}'.format(
                new_repo_info, force_repo, contents_folder)

            print(cmd)

            return_code = os.system(cmd)

            return (return_code == 0), msg

        except Exception as err:
            #logger.error(' Error encountered while creating new repo {0}'.format(new_repo_info))

            msg = str(err)
            #logger.info("exception {0}".format(msg))
            print(msg)
            traceback_msg = traceback.format_exc()
            print(traceback_msg)
            # logger.info(traceback_msg)
            sys_exec_info_msg = sys.exc_info()[2]
            # logger.info(sys_exec_info_msg)
            print(sys_exec_info_msg)
            return False, msg

        return False, msg
