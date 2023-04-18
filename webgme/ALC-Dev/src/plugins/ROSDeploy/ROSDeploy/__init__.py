"""
This is where the implementation of the plugin code goes.
The ROSDeploy-class is imported from both run_plugin.py and run_debug.py
"""

import sys
import traceback
import logging
import json
import re
import time
import os
from pathlib import Path
from webgme_bindings import PluginBase
from future.utils import iteritems
import git
from git import Repo

from alc_utils.setup_repo import RepoSetup

# Setup a logger
logger = logging.getLogger('ROSDeploy')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)  # By default it logs to stderr..
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

cmds = {}
cmds['init_workspace'] = '''bash -c ". /opt/ros/kinetic/setup.bash; mkdir -p /alc_workspace/ros/<<PROJECT_NAME>>/src; cd /alc_workspace/ros/<<PROJECT_NAME>>/src; catkin_init_workspace"'''
cmds['create_package'] = '''bash -c ". /opt/ros/kinetic/setup.bash; cd /alc_workspace/ros/<<PROJECT_NAME>>/src; catkin_create_pkg <<PACKAGE_NAME>> rospy std_msgs; mkdir -p <<PACKAGE_NAME>>/nodes; mkdir -p <<PACKAGE_NAME>>/launch"'''
cmds['make_package'] = '''bash -c ". /opt/ros/kinetic/setup.bash; cd /alc_workspace/ros/<<PROJECT_NAME>>; catkin build"'''
#path when repo is set
cmds['default_repo_relative_path'] = 'alc_gen_ros' #otherwise user specified.
cmds['default_repo_init_workspace'] = '''bash -c ". /opt/ros/kinetic/setup.bash; mkdir -p <<REPO_ROOT>>/<<WORKSPACE_PATH>>/src; cd <<REPO_ROOT>>/<<WORKSPACE_PATH>>/src; catkin_init_workspace"'''
cmds['default_repo_create_package'] = '''bash -c ". /opt/ros/kinetic/setup.bash; cd <<REPO_ROOT>>/<<WORKSPACE_PATH>>/src; catkin_create_pkg <<PACKAGE_NAME>> rospy std_msgs; mkdir -p <<PACKAGE_NAME>>/nodes; mkdir -p <<PACKAGE_NAME>>/launch"'''
cmds['default_repo_make_package'] = '''bash -c ". /opt/ros/kinetic/setup.bash; cd <<REPO_ROOT>>/<<WORKSPACE_PATH>>; catkin build"'''

#check if the following folders are present when user specified
#workspace_path=<<REPO_ROOT>>/<<WORKSPACE_PATH>>
#workspace_path_src=<<REPO_ROOT>>/<<WORKSPACE_PATH>>/src
#workspace_path_package=<<REPO_ROOT>>/<<WORKSPACE_PATH>>/src/<<PACKAGE_NAME>>
#workspace_path_package_nodes=<<REPO_ROOT>>/<<WORKSPACE_PATH>>/src/<<PACKAGE_NAME>>/nodes
#workspace_path_package_launch=<<REPO_ROOT>>/<<WORKSPACE_PATH>>/src/<<PACKAGE_NAME>>/launch

#clone repo
#check for the paths, 
#   if they don't exist, create workspace and/or package and/or nodes/launch and run the relevant catkin commands (for workspace init or create-package)
#copy files and 
#build workspace

#for btree follow similar style. 
#add launch file and codes
#for btree, it would be best to store the .bt file as well.


# get_member_paths(node, name)
# get_parent(node)[source]
# get_path(node)[source]
# get_pointer_path(node, name)[source]
# load_children(node)[source]
# load_members(node, set_name)[source]
# load_pointer(node, pointer_name)[source]
# get_attribute(node, name)[source]
# set_attribute(node, name, value)[source]
# set_pointer(node, name, target)[source]
# add_member(node, name, member)[source]
# del_member(node, name, path)[source]
# del_pointer(node, name)[source]
# delete_node(node)[source]
alc_working_dir_env_var_name = "ALC_WORKING_DIR"
alc_meta_type_name  = "ALC"
build_dir_name      = '.build'


class ROSDeploy(PluginBase):
    #def __init__(self, *args, **kwargs):
    #    super(ROSDeploy, self).__init__(*args, **kwargs)
    def __init__(
            self,
            webgme, commit_hash, BRANCH_NAME, ACTIVE_NODE_PATH, ACTIVE_SELECTION_PATHS, NAMESPACE, USERID,
            config=None, **kwargs
    ):
        PluginBase.__init__(self, webgme, commit_hash, BRANCH_NAME, ACTIVE_NODE_PATH, ACTIVE_SELECTION_PATHS, NAMESPACE)
        self.userid = USERID
        self.cmds = cmds
        self.projectName = ''
        self.alc_node = ''
        self.repo =''
        self.branch = ''
        self.tag = ''
        self.repo_root = ''
        self.config = self.get_current_config()
        self.repo_relative_path = self.config.get('relative_repo_path',cmds['default_repo_relative_path'])
        
    def get_project_name(self):
        projInfo = self.project.get_project_info()
        pname = projInfo['_id']
        self.projectName = pname.replace('+', '_')
    
    def get_ALC_node (self):
        child_node_list = self.core.load_children(self.root_node)
        for child_node in child_node_list:
            if not child_node:
                continue

            child_node_meta_type = self.core.get_meta_type(child_node)
            child_node_meta_type_name = self.core.get_fully_qualified_name(child_node_meta_type)
            if child_node_meta_type_name.endswith(alc_meta_type_name):
                self.alc_node = child_node
                break
    
    def get_repo_info (self):
        if (self.alc_node == ''):
            self.get_ALC_node()
        
        if (self.alc_node == ''):
            raise RuntimeError("Unable to access ALC node")
        
        self.repo = self.core.get_attribute(self.alc_node,'repo')
        self.branch = self.core.get_attribute(self.alc_node,'branch')
        self.tag = self.core.get_attribute(self.alc_node,'tag')
        if (self.repo):
            self.setup_repo()

    def create_repo_root_dir(self):
        alc_working_dir_name = os.environ.get(alc_working_dir_env_var_name, None)
        if alc_working_dir_name:
            self.repo_root = Path(alc_working_dir_name, build_dir_name, self.projectName)
            if  not self.repo_root.exists():
                self.repo_root.mkdir(parents=True)
        else:
            raise RuntimeError("Environment Variable: \"{0}\" is not found.".format(alc_working_dir_env_var_name))

    def setup_repo(self):
        try:
            
            self.create_repo_root_dir()
            logger.info(' trying to setup repo at {0}'.format(str(self.repo_root)))
            if self.repo_root.exists():
                r =  RepoSetup()
                dst_folder_path = os.path.join(str(self.repo_root),self.repo)
                r.clone_repo(self.repo_root, self.repo, self.branch, self.tag, logger)
                self.repo_root = Path(dst_folder_path)
        except Exception as err:
            logger.error(' Error encountered while setting up the repo')
            msg = str(err)
            logger.info("exception {0}".format(msg))
            traceback_msg = traceback.format_exc()
            logger.info(traceback_msg)
            sys_exec_info_msg = sys.exc_info()[2]
            logger.info(sys_exec_info_msg)
            raise RuntimeError("Unable to setup repo")


    def add_to_repo(self):
        if (not self.repo):
            return
        try:
            if self.repo_root.exists():
                r =  RepoSetup()
                r.add_to_repo(self.repo_root,'commited files from rosgen', self.userid, logger)
        except Exception as err:
            logger.error(' Error encountered while commiting to the repo ')
            msg = str(err)
            logger.info("exception {0}".format(msg))
            traceback_msg = traceback.format_exc()
            logger.info(traceback_msg)
            sys_exec_info_msg = sys.exc_info()[2]
            logger.info(sys_exec_info_msg)
            raise RuntimeError("Unable to commit and push to  repo")

    def init_workspace_no_repo(self):
        cmd = self.cmds['init_workspace']
        cmd = re.sub('<<PROJECT_NAME>>', self.projectName, cmd)
        logger.info('init_workspace cmd "{0}"'.format(cmd))
        out = os.system(cmd)
        logger.info('init_workspace result "{0}"'.format(str(out)))

    def init_workspace_repo(self, repo_root, repo_relative_path):
        workspace_path = Path(repo_root,repo_relative_path);
        if (not workspace_path.exists()):
            cmd = self.cmds['default_repo_init_workspace']
            cmd = re.sub('<<REPO_ROOT>>', repo_root, cmd)
            cmd = re.sub('<<WORKSPACE_PATH>>', repo_relative_path, cmd)
            logger.info('init_workspace cmd "{0}"'.format(cmd))
            out = os.system(cmd)
            logger.info('init_workspace result "{0}"'.format(str(out)))
        else:
            logger.info('init_workspace - workspace  "{0}"  already exists'.format(str(workspace_path)))
        
    def init_workspace(self):
        if (self.repo == ''):
            return self.init_workspace_no_repo()
        
        if (self.repo_root.exists()):
            if (not self.repo_relative_path):
                self.repo_relative_path = cmds['default_repo_relative_path']
            else:
                if (self.repo_relative_path.endswith('/src')):
                    self.repo_relative_path = self.repo_relative_path[:-4]
            return self.init_workspace_repo(str(self.repo_root), self.repo_relative_path)
        else:
            raise RuntimeError("Repo-root path  (%s) does not exist (init workspace package) " % str(self.repo_root))


    def create_package_no_repo(self, packageName):
        cmd = self.cmds['create_package']
        cmd = re.sub('<<PROJECT_NAME>>', self.projectName, cmd)
        cmd = re.sub('<<PACKAGE_NAME>>', packageName.lower(), cmd)
        os.system(cmd)    

    def create_package_repo(self, repo_root, repo_relative_path, packageName):
        cmd = self.cmds['default_repo_create_package']
        cmd = re.sub('<<REPO_ROOT>>', repo_root, cmd)
        cmd = re.sub('<<WORKSPACE_PATH>>', repo_relative_path, cmd)
        cmd = re.sub('<<PACKAGE_NAME>>', packageName.lower(), cmd)
        os.system(cmd)

    

    def create_package(self, packageName):
        if (self.repo == ''):
            return self.create_package_no_repo(packageName)

        package_path = Path(str(self.repo_root),self.repo_relative_path, 'src', packageName.lower())
        if (not package_path.exists()):
            return self.create_package_repo(str(self.repo_root), self.repo_relative_path,packageName)
        else:
            logger.info('package "{0}" already exists. package-path:  "{1}"'.format(packageName.lower(),str(package_path)))
            nodes_path = Path(str(self.repo_root),self.repo_relative_path, 'src', packageName.lower(),'nodes')
            if (not nodes_path.exists()):
                nodes_path.mkdir(parents=True)
            launch_path = Path(str(self.repo_root),self.repo_relative_path, 'src', packageName.lower(),'launch')
            if (not launch_path.exists()):
                launch_path.mkdir(parents=True)


    def build_no_repo(self):
        cmd = self.cmds['make_package']
        cmd = re.sub('<<PROJECT_NAME>>', self.projectName, cmd)
        os.system(cmd)

    def build_repo(self,repo_root, repo_relative_path):
        cmd = self.cmds['default_repo_make_package']
        cmd = re.sub('<<REPO_ROOT>>', repo_root, cmd)
        cmd = re.sub('<<WORKSPACE_PATH>>', repo_relative_path, cmd)
        os.system(cmd)

    def build(self):
        if (self.repo == ''):
            return self.build_no_repo()
        workspace_path = Path(str(self.repo_root),self.repo_relative_path)
        if (workspace_path.exists()):
            self.build_repo(str(self.repo_root),self.repo_relative_path)
        else:
            raise RuntimeError("Build Failed. Workspace-Path  (%s) does not exist. " % str(workspace_path))

    def create_file(self, filepath, contents, isExecutable=False):
        f = open(filepath, "w")
        f.write(contents)
        f.close()
        if (isExecutable):
            mode = os.stat(filepath).st_mode
            mode |= (mode & 0o444) >> 2  # copy R bits to X
            os.chmod(filepath, mode)

    def copy_srcs(self, packageName, compName, ros_info):
        code_compimpl = self.core.get_attribute(ros_info, 'srcComponentImpl')
        launch_info = self.core.get_attribute(ros_info, 'launchInfo')

        btree_model = self.core.get_attribute(ros_info, 'BTreeModel')
        btree_gen_code = {}
        if (btree_model):
            childnodes = self.core.load_children(ros_info)
            for c in childnodes:
                if self.core.get_meta_type(c) == self.META["ALCMeta.Code"]:
                    name = self.core.get_attribute(c, 'name')
                    code = self.core.get_attribute(c, 'code')
                    if (name and code):
                        btree_gen_code[name]= code

        
        alc_working_dir = os.getenv('ALC_WORKING_DIR')
        package_path_no_repo = os.path.join(alc_working_dir, 'ros', self.projectName, 'src', packageName)
        package_path = package_path_no_repo

        if (self.repo):
            package_path_repo = Path(str(self.repo_root),self.repo_relative_path, 'src', packageName.lower())
            package_path = str(package_path_repo)

        if not os.path.exists(package_path):
            raise RuntimeError("Package path  (%s) does not exist (after creating package) " % package_path)
        src_path = os.path.join(package_path, 'nodes')
        if not os.path.exists(src_path):
            raise RuntimeError("Package  source path  (%s) does not exist (after creating package) " % src_path)
        launch_path = os.path.join(package_path, 'launch')
        if not os.path.exists(launch_path):
            raise RuntimeError("Package  launch path  (%s) does not exist (after creating package) " % launch_path)

        comp_impl_path = os.path.join(src_path, compName + "_impl.py")
        self.create_file(comp_impl_path, code_compimpl, True)

        if (btree_model):
            btree_model_file = os.path.join(src_path, compName + "_btree.bt")
            self.create_file(btree_model_file, btree_model)
            btkeys = btree_gen_code.keys()
            for b in btkeys:
                file_path = os.path.join(src_path, b)
                self.create_file(file_path, btree_gen_code[b])
        if launch_info:
            comp_launch_path = os.path.join(launch_path, "start_" + compName + ".launch")
            self.create_file(comp_launch_path, launch_info)

    def getBlockPackageName(self):
        ret = ''
        parent = self.core.get_parent(self.active_node)
        while True:
            if (not parent):
                break
            if self.core.get_meta_type(parent) == self.META["ALCMeta.BlockPackage"]:
                ret = self.core.get_attribute(parent, 'name')
                break
            if (not (self.core.get_meta_type(parent) == self.META["ALCMeta.Block"])):
                break
            parent = self.core.get_parent(parent)
        return ret

    def deployBlock(self, packageName, node, ros_info):
        compName = self.core.get_attribute(node, 'name')
        compName = re.sub(' ', '_', compName)
        self.copy_srcs(packageName, compName, ros_info)

    def main(self):
        try:
            core = self.core
            root_node = self.root_node
            active_node = self.active_node
            self.get_project_name()
            if (self.core.get_meta_type(active_node) == self.META["ALCMeta.Block"]):
                active_node_name = self.core.get_attribute(active_node, 'name')
                role = core.get_attribute(active_node, 'Role')
                if (role != 'Node' and role != 'Driver'):
                    self.result_set_success(False)
                    self.create_message(self.active_node,
                                        'Selected node must be a Block-Package or a Block with role=Node or role=Driver ',
                                        'error')
                    self.result_set_error("Selected Block's role attribute needs to be Node or Driver")
                    exit()
                cnodes = self.core.load_children(self.active_node)
                ros_info = ''
                for c in cnodes:
                    if self.core.get_meta_type(c) == self.META["ALCMeta.ROSInfo"]:
                        ros_info = c
                        break
                if not ros_info:
                    self.result_set_success(False)
                    self.result_set_error("No ROSInfo node found in (%s)" % active_node_name)
                    exit()
                packageName = self.core.get_attribute(ros_info, 'Package')
                if not packageName:
                    packageName = self.getBlockPackageName()
                    if (not packageName):
                        self.result_set_success(False)
                        self.result_set_error("Package name is not specified in the ROSInfo node")
                        exit()

                self.get_repo_info()
                self.init_workspace()
                self.create_package(packageName)
                self.deployBlock(packageName, active_node, ros_info)
                self.add_to_repo()
                if not self.repo:
                    self.build()
                self.result_set_success(True)
                commit_info = self.util.save(root_node, self.commit_hash, 'master', 'ros code deployed')
                logger.info('committed :{0}'.format(commit_info))

            elif self.core.get_meta_type(active_node) == self.META["ALCMeta.BlockPackage"]:
                packageName = self.core.get_attribute(active_node, 'name')
                self.get_repo_info()
                self.init_workspace()
                self.create_package(packageName)

                allchildren = self.core.load_sub_tree(active_node)
                for x in allchildren:
                    if self.core.get_meta_type(x) != self.META["ALCMeta.ROSInfo"]:
                        continue
                    base_x = self.core.get_base(x)
                    if not self.core.is_meta_node(base_x):
                        continue
                    block = self.core.get_parent(x)
                    role = self.core.get_attribute(block, 'Role')
                    if role == 'Node' or role == 'Driver':
                        self.deployBlock(packageName, block, x)

                self.add_to_repo()
                if not self.repo:
                    self.build()
                self.result_set_success(True)
                commit_info = self.util.save(self.root_node, self.commit_hash, 'master', 'ros code deployed')
                logger.info('committed :{0}'.format(commit_info))

            else:
                self.create_message(self.active_node, 'Selected node must be a Block-Package or a Block with role=Node '
                                                      'or role=Driver ', 'error')
                self.result_set_error('Selected node must be a Block-Package or a Block with role=Node or role=Driver ')
                self.result_set_success(False)
                exit()
        except Exception as err:
            #self.send_notification(str(err))
            #raise err
            # msg = str(err)
            # self.create_message(self.active_node, msg, 'error')
            # self.result_set_error('ROSDeploy: Error encoutered.  Check result details.')
            # self.result_set_success(False)
            # exit()
            msg = str(err)
            logger.info("exception {0}".format(msg))
            traceback_msg = traceback.format_exc()
            logger.info(traceback_msg)
            sys_exec_info_msg = sys.exc_info()[2]
            logger.info(sys_exec_info_msg)
            self.create_message(self.active_node, msg, 'error')
            self.create_message(self.active_node, traceback_msg, 'error')
            self.create_message(self.active_node, str(sys_exec_info_msg), 'error')
            self.result_set_error('ROSDeploy Plugin: Error encountered.  Check result details.')
            self.result_set_success(False)
            exit()
