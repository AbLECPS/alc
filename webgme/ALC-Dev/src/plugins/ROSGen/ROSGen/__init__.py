"""
This is where the implementation of the plugin code goes.
The ROSGen-class is imported from both run_plugin.py and run_debug.py
"""
import sys
import traceback
import logging
import os
from webgme_bindings import PluginBase
import re
import string
import jinja2
from alc_utils.textx.src import btree_codegen
# Setup a logger
logger = logging.getLogger('ROSGen')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)  # By default it logs to stderr..
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

_package_directory = os.path.dirname(__file__)


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


#for btree use the btree-attribute (.bt) file to generate the code.
# the code generation invokation and storing of the files will need to be altered to allow multiple code generatton and storing in the model.



class ROSGen(PluginBase):
    def __init__(self, *args, **kwargs):
        super(ROSGen, self).__init__(*args, **kwargs)

        self.ros_interface_info = {}
        self.interface_keys = ['Publisher', 'Subscriber', 'Service Server', 'Service Client', 'Action Server',
                               'Action Client']
        for i in self.interface_keys:
            self.ros_interface_info[i] = []

        self.ros_info = {}
        self.msg_imports = []
        self.ros_info_keys = ['compName', 'pkgName', 'namespace', 'nodeFrequency']
        for i in self.ros_info_keys:
            self.ros_info[i] = ''

        self.ros_params = []
        self.ros_args = []

        self.pub_info = {}
        self.sub_info = {}
        self.pubInfoKeys = []
        self.subInfoKeys = []
        self.rosInfoNode = None
        self.pcode_compimpl = {}
        self.pcode_rosimpl = {}
        self.code_compimpl = ''
        self.code_rosimpl = ''
        self.lec_info = {}
        self.lec_info['use'] = 0
        self.linked_port = {}
        self.counter = 0

    def updateName(self, name):
        name = re.sub(' ', '_', name)
        return name

    def getCodeFromAttributes(self):
        if self.rosInfoNode:
            self.code_compimpl = self.core.get_attribute(self.rosInfoNode, 'srcComponentImpl')
            self.code_rosimpl = self.core.get_attribute(self.rosInfoNode, 'srcROSImpl')

    def gatherUserUpdates(self):
        self.pcode_compimpl = self.getUserUpdates(self.code_compimpl)
        self.pcode_rosimpl = self.getUserUpdates(self.code_rosimpl)

    def getUserUpdates(self, code):
        if (not code):
            return {}
        codelist = code.splitlines()
        cur_key = ''
        lines = []
        contents = {}
        for l in codelist:
            x = l.strip()
            if (x.startswith('# protected region') or x.startswith('#********** protected region')):
                if (not cur_key and (x.endswith('begin #') or x.endswith('begin **********#'))):
                    cur_key = x
                    cur_key = cur_key.replace('#**********', '#')
                    cur_key = cur_key.replace('**********#', '#')
                    continue
                if (cur_key and (
                        x.endswith('end #') or x.endswith('end **********#') or x.endswith('end   **********#'))):
                    if (len(lines)):
                        contents[cur_key] = '\n'.join(lines, '\n')
                    lines = []
                    cur_key = ''
                    continue;
            if (cur_key):
                lines.append(l)
        return contents

    def getUserUpdatesForROSImpl(self):
        if (not self.code_rosimpl):
            return

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

    def processLECInfo(self, lec):
        self.lec_info['use'] = 1
        self.lec_info['deployment_key'] = self.core.get_attribute(lec, 'DeploymentKey')
        self.lec_info['category'] = self.core.get_attribute(lec, 'Category')
        self.lec_info['name'] = self.core.get_attribute(lec, 'name')

    def processROSInfo(self, rosinfo):
        # self.ros_info['pkgName'] = self.core.get_attribute(rosinfo,'Package')
        # if (self.ros_info['pkgName']==''):
        #    self.ros_info['pkgName'] = self.getBlockPackageName()
        #    self.core.set_attribute(rosinfo,'Package', self.ros_info['pkgName'])
        self.ros_info['pkgName'] = self.getBlockPackageName()
        if (self.ros_info['pkgName']):
            self.core.set_attribute(rosinfo, 'Package', self.ros_info['pkgName'])

        self.ros_info['namespace'] = self.core.get_attribute(rosinfo, 'Namespace')
        self.ros_info['nodeFrequency'] = self.core.get_attribute(rosinfo, 'UpdateFrequency')
        self.ros_info['compName'] = self.core.get_attribute(self.active_node, 'name')
        self.ros_info['compName'] = self.updateName(self.ros_info['compName'])
        self.ros_info['Type'] = self.core.get_attribute(rosinfo, 'Type')
        #if (self.ros_info['Type'] == ''):
        self.ros_info['Type'] = self.ros_info['compName'] + "_impl.py"
        self.core.set_attribute(rosinfo, 'Type', self.ros_info['Type'])

        launchFileLocation = os.path.join('$(find ' + self.ros_info['pkgName'] + ')', 'launch',
                                          "start_" + self.ros_info['compName'] + ".launch")
        self.core.set_attribute(rosinfo, 'LaunchFileLocation', launchFileLocation)

        self.processParams(rosinfo)

    def processParams(self, rosinfo):
        cnodes = self.core.load_children(rosinfo)
        for c in cnodes:
            # if self.core.get_meta_type(c) == self.META["ALCMeta.ROSParam"]:
            #    name = self.core.get_attribute(c,'name')
            #    name = self.updateName(name)
            #    value = self.core.get_attribute(c,'default')
            #    self.ros_params.append((name,value))
            if self.core.get_meta_type(c) == self.META["ALCMeta.ROSArgument"]:
                name = self.core.get_attribute(c, 'name')
                name = self.updateName(name)
                value_default = self.core.get_attribute(c, 'default')
                value = value_default
                reqd = False
                linked_port = self.core.get_pointer_path(c, 'linked_port')
                if (linked_port):
                    self.linked_port[linked_port] = c
                    self.counter += 1
                    continue
                self.ros_params.append((name, value, reqd))

    def create_child_param(self, port):
        nb = self.core.create_child(self.rosInfoNode, self.META["ALCMeta.ROSArgument"])
        pos = {}
        pos['x'] = 100
        pos['y'] = 100 + 25 * (self.counter - 1)
        name = 'topic_'
        ptype = self.core.get_attribute(port, 'PortType')
        if (ptype == 'Subscriber'):
            name += 'sub_'
        if (ptype == 'Publisher'):
            name += 'pub_'
        if (ptype == 'Service Server'):
            name += 'service_'
        if (ptype == 'Service Client'):
            name += 'client_'
        if (ptype == 'Action Server'):
            name += 'as_'
        if (ptype == 'Action Client'):
            name += 'ac_'
        name += self.updateName(self.core.get_attribute(port, 'name'))
        self.core.set_registry(nb, 'position', pos)
        self.core.set_attribute(nb, 'name', name)
        self.core.set_pointer(nb, 'linked_port', port)
        self.counter += 1

    def processPort(self, p):
        ptype = self.core.get_attribute(p, 'PortType')
        if (ptype.upper() == 'OTHER'):
            return
        linked_keys = self.linked_port.keys()
        port_path = self.core.get_path(p)
        if (port_path not in linked_keys):
            self.create_child_param(p)

        msg_obj = self.core.load_pointer(p, 'messagetype')
        cls_type = ''
        pkg_type = ''
        if (msg_obj):
            cls_type = self.core.get_attribute(msg_obj, 'name')
            cls_type = self.updateName(cls_type)
            pkg_obj = self.core.get_parent(msg_obj)
            if (pkg_obj):
                pkg_type = self.core.get_attribute(pkg_obj, 'name')
                pkg_type = self.updateName(pkg_type)
        name = self.core.get_attribute(p, 'name')
        name = self.updateName(name)
        topic = self.core.get_attribute(p, 'Topic')
        if (topic):
            if topic[0] in ['\'', '\"']:
                topic = topic[1:]
            if topic[-1] in ['\'', '\"']:
                topic = topic[:-1]
        

        # FIXME: Why is port only added to ros_interface info if pkg_type and cls_type are defined?
        #        This can cause launch files to be generated without all the port topic parameters.
        #        Should execution be aborted if this happens to prevent invalid files from being generated?
        if pkg_type and cls_type:
            self.ros_interface_info[ptype].append((pkg_type, cls_type, name, topic))

            if (pkg_type, cls_type) not in self.msg_imports:
                self.msg_imports.append((pkg_type, cls_type))
        else:
            msg = {"message": "Failed to determine message type for port \"%s\"." % self.core.get_attribute(p, 'name'),
                   "severity": "error"}
            self.send_notification(msg)

        if pkg_type and cls_type and ptype in ['Publisher', 'Subscriber']:
            queue_size = self.core.get_attribute(p, 'queueSize')
            ps_info = {}
            ps_info['name'] = name
            ps_info['topic'] = topic
            ps_info['type'] = cls_type
            ps_info['queue_size'] = queue_size
            if (ptype == 'Publisher'):
                self.pub_info[name] = ps_info
            if (ptype == 'Subscriber'):
                self.sub_info[name] = ps_info

    def gatherInfo(self):
        cnodes = self.core.load_children(self.active_node)

        # Find ROS Info object, or create one if none exists
        for c in cnodes:
            if self.core.get_meta_type(c) == self.META["ALCMeta.ROSInfo"]:
                self.rosInfoNode = c
        if self.rosInfoNode is None:
            self.send_notification("No ROS Info object detected, will create default object.")
            self.rosInfoNode = self.core.create_child(self.active_node, self.META["ALCMeta.ROSInfo"])
            self.core.set_registry(self.rosInfoNode, 'position', {"x": 100, "y": 100})
        self.processROSInfo(self.rosInfoNode)

        for c in cnodes:
            if self.core.get_meta_type(c) == self.META["ALCMeta.SignalPort"]:
                self.processPort(c)

        cnodes = self.core.load_sub_tree(self.active_node)
        for c in cnodes:
            if self.core.get_meta_type(c) == self.META["ALCMeta.LEC_Model"]:
                self.processLECInfo(c)
        self.sort_info()
        self.getCodeFromAttributes()
        self.gatherUserUpdates()

    def sort_info(self):
        self.pubInfoKeys = self.pub_info.keys()
        self.pubInfoKeys.sort()
        self.subInfoKeys = self.sub_info.keys()
        self.subInfoKeys.sort()
        self.ros_interface_info['Subscriber'].sort(key=lambda x: x[2])
        self.ros_interface_info['Publisher'].sort(key=lambda x: x[2])
        self.ros_interface_info['Service Server'].sort(key=lambda x: x[2])
        self.ros_interface_info['Service Client'].sort(key=lambda x: x[2])
        self.ros_interface_info['Action Server'].sort(key=lambda x: x[2])
        self.ros_interface_info['Action Client'].sort(key=lambda x: x[2])
        self.ros_params.sort(key=lambda x: x[0])
        # self.ros_args.sort(key=lambda x:x[0])
    
    def generate_and_update_btree_code(self,template_loader,template_env):
        if not self.rosInfoNode:
            return
        btree_model  = self.core.get_attribute(self.rosInfoNode, 'BTreeModel')
        if (not btree_model):
            return
        childnodes = self.core.load_children(self.rosInfoNode)
        btree_gen_code = {}
        codes = []
        for c in childnodes:
            if self.core.get_meta_type(c) == self.META["ALCMeta.Code"]:
                name = self.core.get_attribute(c, 'name')
                code = self.core.get_attribute(c, 'code')
                if (name and code):
                    codes.append(c)
                    btree_gen_code[name]= code
        btree_main_code, updated_btree_gen_code = btree_codegen.run(btree_model, btree_gen_code)
        self.core.set_attribute(self.rosInfoNode, 'srcComponentImpl', btree_main_code)

        updated = []
        not_available = []
        pos={}
        pos['x']=100
        pos['y']=100
        counter = 0
                        
        for c in codes:
            name = self.core.get_attribute(c, 'name')
            if (name in updated_btree_gen_code):
                self.core.set_attribute(c, 'code', updated_btree_gen_code[name])
                counter +=1
                pos['y'] = 100*counter
                self.core.set_registry(c, 'position', pos)
                updated.append(name)
            else:
                not_available.append(c)

        namekeys = updated_btree_gen_code.keys()
        for name in namekeys:
            if (name in updated):
                continue
            c1 = self.core.create_child(self.rosInfoNode, self.META["ALCMeta.Code"])
            self.core.set_attribute(c1, 'name', name)
            self.core.set_attribute(c1, 'code', updated_btree_gen_code[name])
            counter +=1
            pos['y'] = 100*counter
            self.core.set_registry(c1, 'position', pos)
        
        while not_available:
            x = not_available.pop()
            self.core.delete_node(x)
        
        generated_launch_info = self.generate_ros_launch(template_loader,template_env)
        self.core.set_attribute(self.rosInfoNode, 'launchInfo', generated_launch_info)
        commit_info = self.util.save(self.root_node, self.commit_hash, 'master', 'ros btree code generated')
    
    def generate_ros_comp(self, template_loader, template_env):
        comp_template = template_env.get_template("ros_comp_impl.template")
        generated_comp_impl = comp_template.render(packageName=self.ros_info['pkgName'],
                                                componentName=self.ros_info['compName'],
                                                msgimports=self.msg_imports,
                                                params=self.ros_params,
                                                subscribers=self.ros_interface_info['Subscriber'],
                                                publishers=self.ros_interface_info['Publisher'],
                                                serviceServers=self.ros_interface_info['Service Server'],
                                                serviceClients=self.ros_interface_info['Service Client'],
                                                actionServers=self.ros_interface_info['Action Server'],
                                                actionClients=self.ros_interface_info['Action Client'],
                                                pubInfoKeys=self.pubInfoKeys,
                                                subInfoKeys=self.subInfoKeys,
                                                pubInfo=self.pub_info,
                                                subInfo=self.sub_info,
                                                componentFrequency=self.ros_info['nodeFrequency'],
                                                pcode=self.pcode_compimpl,
                                                lec_info=self.lec_info)
        logger.info('*************************************')
        logger.info('*************************************')
        logger.info('generated_comp_impl "{0}" '.format(generated_comp_impl))
        logger.info('*************************************')
        logger.info('*************************************')
        return generated_comp_impl

    def generate_ros_launch(self, template_loader, template_env):
        launch_template = template_env.get_template("ros_launch.template")
        generated_launch_info = launch_template.render(node_info=self.ros_info,
                                                    ros_params=self.ros_params,
                                                    ros_args=self.ros_params,
                                                    subscribers=self.ros_interface_info['Subscriber'],
                                                    publishers=self.ros_interface_info['Publisher'],
                                                    serviceServers=self.ros_interface_info['Service Server'],
                                                    serviceClients=self.ros_interface_info['Service Client'],
                                                    actionServers=self.ros_interface_info['Action Server'],
                                                    actionClients=self.ros_interface_info['Action Client'],
                                                    pubInfoKeys=self.pubInfoKeys,
                                                    subInfoKeys=self.subInfoKeys,
                                                    pubInfo=self.pub_info,
                                                    subInfo=self.sub_info,
                                                    lec_info=self.lec_info)
        logger.info('*************************************')
        logger.info('*************************************')
        logger.info('generated launch info "{0}" '.format(generated_launch_info))
        logger.info('*************************************')
        logger.info('*************************************')

        return generated_launch_info

    def main(self):
        try:
            core = self.core
            root_node = self.root_node
            active_node = self.active_node
            name = core.get_attribute(active_node, 'name')
            role = core.get_attribute(active_node, 'Role')
            if role != 'Node' and role != 'Driver':
                raise RuntimeError("Selected Block's role attribute needs to be Node or Driver")

            base_node = self.core.get_base(active_node)
            if not self.core.is_meta_node(base_node):
                raise RuntimeError("ROS Gen should be invoked on base models in the Block Library")


            self.gatherInfo()

            if (self.rosInfoNode):
                btree_model  = self.core.get_attribute(self.rosInfoNode, 'BTreeModel')
                template_loader = jinja2.FileSystemLoader(searchpath=_package_directory)
                template_env = jinja2.Environment(loader=template_loader, trim_blocks=True, lstrip_blocks=True)
                if (not btree_model):
                    generated_comp_impl = self.generate_ros_comp(template_loader,template_env)
                    generated_launch_info = self.generate_ros_launch(template_loader,template_env)
                    self.core.set_attribute(self.rosInfoNode, 'srcComponentImpl', generated_comp_impl)
                    self.core.set_attribute(self.rosInfoNode, 'launchInfo', generated_launch_info)
                    commit_info = self.util.save(root_node, self.commit_hash, 'master', 'ros code generated')
                else:
                    self.generate_and_update_btree_code(template_loader,template_env)

            self.result_set_success(True)
        except Exception as err:
            msg = str(err)
            logger.info("exception {0}".format(msg))
            traceback_msg = traceback.format_exc()
            logger.info(traceback_msg)
            sys_exec_info_msg = sys.exc_info()[2]
            logger.info(sys_exec_info_msg)
            self.create_message(self.active_node, msg, 'error')
            self.create_message(self.active_node, traceback_msg, 'error')
            self.create_message(self.active_node, str(sys_exec_info_msg), 'error')
            self.result_set_error('ROSGen Plugin: Error encountered.  Check result details.')
            self.result_set_success(False)
            exit()
