"""
This is where the implementation of the plugin code goes.
The ActivityInit-class is imported from both run_plugin.py and run_debug.py
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
from urllib.parse import urlunsplit, urljoin
import urllib.request
from alc_utils.slurm_executor import WebGMEKeys
from alc_utils.setup_repo import RepoSetup

from ActivityInit.communication import *

# Setup a logger
logger = logging.getLogger('ActivityInit')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)  # By default it logs to stderr..
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

url_hostname = "localhost"

POS_X_INPUT = 100
POS_X_CONTEXT = 400
POS_X_PARAMETER = 700
POS_X_OUTPUT= 1000
POS_Y_START = 100
POS_Y_INCREMENT = 100

activity_definition_folder_root_name = "activity_definitions"
model_folder_root_name               = 'model'
alc_working_dir_env_var_name         = "ALC_WORKING_DIR"
exec_dir_name                        = '.exec'
alc_meta_type_name                   = "ALC"
definition_filename                  = "Definition.json"
alc_core_repo                        = "alc_core"


class ActivityInit(PluginBase):

    def setup_communication(self, communication_file):
        set_communication_file(communication_file)

    def __init__(
            self,
            webgme, commit_hash, BRANCH_NAME, ACTIVE_NODE_PATH, ACTIVE_SELECTION_PATHS, NAMESPACE,
            config=None, **kwargs
    ):
        PluginBase.__init__(self, webgme, commit_hash, BRANCH_NAME, ACTIVE_NODE_PATH, ACTIVE_SELECTION_PATHS, NAMESPACE)

        webgme_port = kwargs.pop(WebGMEKeys.webgme_port_key, 8888)

        self.metadata_url = urlunsplit(
            ['http', "{0}:{1}".format(url_hostname, webgme_port), "/rest/blob/metadata/", None, None]
        )
        self.download_url = urlunsplit(
            ['http', "{0}:{1}".format(url_hostname, webgme_port), "/rest/blob/download/", None, None]
        )

        if not config:
            self.config = self.get_current_config()
        else:
            self.config = config
        
        self.activity_definition = ''
        self.activity_json =  ''
        self.category = ''
        self.name     =  ''
        self.contexts = {}
        self.inputs = {}
        self.outputs = {}
        self.parameters = {}
        self.alc_node = ''
        self.repo =''
        self.branch = ''
        self.tag = ''
        self.repo_root = ''
        self.alc_repo_root =''
        self.setup_path=''
        self.current_choice = ''
        #self.repo_root = Path(alc_working_dir_name, exec_dir_name, self.repo)

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

    def get_repo_root_dir(self):
        alc_working_dir_name = os.environ.get(alc_working_dir_env_var_name, None)
        if alc_working_dir_name:
            self.repo_root = Path(alc_working_dir_name, exec_dir_name,self.repo)
            self.alc_repo_root = Path(alc_working_dir_name, exec_dir_name,alc_core_repo,alc_core_repo)
            if  not self.repo_root.exists():
                self.repo_root.mkdir(parents=True)
        else:
            raise RuntimeError("Environment Variable: \"{0}\" is not found.".format(alc_working_dir_env_var_name))

    def setup_repo(self):
        try:
            self.get_repo_root_dir()
            logger.info(' trying to setup repo at {0}'.format(str(self.repo_root)))
            if self.repo_root.exists():
                r =  RepoSetup()
                dst_folder_path = os.path.join(str(self.repo_root),self.repo)
                #r.clone_repo(self.repo_root, self.repo, self.branch, self.tag, logger)
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
    
    def get_all_activity_definitions(self):
        self.get_repo_info()
        print('1')
        activity_definitions_root = Path(self.repo_root, model_folder_root_name, activity_definition_folder_root_name)
        initlen = len(str(activity_definitions_root))
        folders = str(activity_definitions_root)
        activities = {}
        for root, dirs, files in os.walk(folders,followlinks=True):
            for file in files:
                if file.endswith(definition_filename):
                    activityname = root[initlen+1:]
                    activities[activityname]= os.path.join(root, file)
        print('finished activities')
        print(str(activities))
        

        alc_activity_definitions_root = Path(self.alc_repo_root, model_folder_root_name, activity_definition_folder_root_name)
        initlen = len(str(alc_activity_definitions_root))
        folders = str(alc_activity_definitions_root)
        for root, dirs, files in os.walk(folders,followlinks=True):
            for file in files:
                if file.endswith(definition_filename):
                    activityname = "alc/"+root[initlen+1:]
                    activities[activityname]= os.path.join(root, file)
        print('finished alc activities')
        print(str(activities))
        return activities


        
        
    
    def get_definition_from_json(self):
        #self.activity_definition = self.config.get('activity', '')
        #self.activity_json = self.config.get('activity_json','')
        #logger.info('activity_definition "{0}" '.format(self.activity_definition))
        #logger.info('activity_json "{0}" '.format(self.activity_json))
            
        if (self.activity_definition):
            return
            
        if (not self.activity_json):
            raise Exception("Activity JSON or Activity Definition is not available")
        
        
        url = urljoin(self.download_url, self.activity_json)
        request = urllib.request.urlopen(url)
        data = request.read()
        self.activity_definition = json.loads(data.decode('utf8'))
        keys  = list(self.activity_definition.keys())
        logger.info('activity_definition keys "{0}" '.format(keys))
        self.name = keys[0]
        self.activity_definition = self.activity_definition[self.name]
        logger.info('activity_definition "{0}" '.format(self.activity_definition))
    
    def get_activity_definition_from_user_choice(self,activity_definitions,user_choice):
        activity_definition_path = activity_definitions[user_choice]
        with open(activity_definition_path) as json_file:
            data = json.load(json_file)
        

        self.activity_definition = data#json.loads(data.decode('utf8'))
        self.setup_path='$REPO_HOME/model/activity_definitions/'+user_choice
        print(user_choice)
        if (user_choice.startswith('alc/')):
            self.setup_path='$ALC_REPO_HOME/model/activity_definitions/'+user_choice[4:]
        
        keys  = list(self.activity_definition.keys())
        logger.info('activity_definition keys "{0}" '.format(keys))
        self.name = keys[0]
        self.activity_definition = self.activity_definition[self.name]
        logger.info('activity_definition "{0}" '.format(self.activity_definition))


    def get_activity_info(self):
        self.category = self.activity_definition.get('Category','')
        self.name     = self.activity_definition.get('Name',self.name)
        self.contexts = self.activity_definition.get('context',{})
        self.inputs = self.activity_definition.get('input',{})
        self.outputs = self.activity_definition.get('output',{})
        self.parameters = self.activity_definition.get('parameters',{})
        self.choices = self.activity_definition.get('Choices','')
        self.choices = list(map(lambda x: x.strip(), self.choices.split(","))) if bool(self.choices) else []
        self.exclusive_choices = self.activity_definition.get('Exclusive',True)
        
    def check_activity_info(self):
        if (not self.activity_definition or not self.name or not self.parameters):
            raise Exception("Activity Source, Definition or Parameters are not set")
    
    

    def initialize(self):
        self.core.set_attribute(self.active_node,'Label', self.name)
        self.core.set_attribute(self.active_node,'Category', self.category )
        self.core.set_attribute(self.active_node,'Definition', json.dumps(self.activity_definition, indent=4, sort_keys=True))
        self.core.set_attribute(self.active_node,'Setup_Folder', self.setup_path)
        self.core.set_attribute(self.active_node,'CurrentChoice', self.current_choice)

    def create_node(self, parent, name, meta, x,counter):
        node = self.core.create_child(parent, meta)
        self.core.set_registry(node, 'position', {'x':x, 'y':POS_Y_START+(counter-1)*POS_Y_INCREMENT})
        self.core.set_attribute(node,'name',name)
        return node
    
    def populate_child_nodes(self, parent,  children, meta):
        num = 0
        for cname, citem in children.items():
            num += 1
            c = self.create_node(parent,cname,meta, POS_X_INPUT, num)
            self.set_attributes(c, citem)
    
    def set_attributes(self,node, attribute_dict):
        alist = self.core.get_attribute_names(node)
        for aname, avalue in attribute_dict.items():
            if aname in alist:
                self.core.set_attribute(node, aname, avalue)

    def populate_items(self, items, meta, posx,child_info):
        num = 0
        for name, contents in items.items():
            num +=1
            node = self.create_node(self.active_node,name, meta, posx, num)
            self.set_attributes(node,contents)
            for child_key, child_meta in child_info:
                children = contents.get(child_key,{})
                self.populate_child_nodes(node, children,child_meta)

    def get_meta_element(self,name):
        meta_name = 'ALC_EP_Meta.'+name
        alternate = name
        meta_element = self.META.get(meta_name,None)
        if (not meta_element):
            meta_element = self.META.get(alternate,None)
        if (not meta_element):
            raise Exception("ActivityInit: Unable to get meta element {0} or {1}".format(meta_name,alternate))
        return meta_element


    
    def populate_model(self):
        self.initialize()
        context_meta = self.get_meta_element('Context')
        content_meta = self.get_meta_element('content')
        paramtable_meta = self.get_meta_element('ParamsTable')
        parameter_meta = self.get_meta_element('Parameter')
        input_meta = self.get_meta_element('Input')
        output_meta = self.get_meta_element('Output')
        field_meta = self.get_meta_element('Field')
        result_meta = self.get_meta_element('Result')

        self.populate_items(self.contexts, context_meta,POS_X_CONTEXT, [('parameters',content_meta)])
        self.populate_items(self.inputs, input_meta,POS_X_INPUT, [])
        self.populate_items(self.outputs, output_meta,POS_X_OUTPUT, [('attributes',field_meta)])
        self.populate_items(self.parameters, paramtable_meta,POS_X_PARAMETER, [('parameters',parameter_meta)])
        self.create_node(self.active_node,'Results',result_meta,POS_X_OUTPUT,4)
        #self.populate_contexts()
        #self.populate_inputs()
        #self.populate_outputs()
        #self.populate_parameters()


    def check_if_filled(self):
        src = self.core.get_attribute(self.active_node,'Label')
        cat = self.core.get_attribute(self.active_node,'Category')
        definition = self.core.get_attribute(self.active_node,'Definition')

        

        if (src or definition):
            logger.info('src "{0}" '.format(src))
            logger.info('cat "{0}" '.format(cat))
            logger.info('definition "{0}" '.format(definition))
            raise Exception("Activity definition has already been populated")
            
        
        return False
    
    def get_user_response(self,choices):
        
        response = ask_user(self, 
                                {
                                    "title": "Activity Choice",
                                    "description":"Please choose activity to initialize",
                                    "fields": [
                                                {
                                                "name": "activity",
                                                "displayName": "Choose an Activity",
                                                "description": "",
                                                "value": "Unknown",
                                                "valueType": "string",
                                                "valueItems": choices
                                                }]
                                   }
                            )
        return response["activity"]

    def get_user_choice_response(self,choices):
        if (len(choices) == 0):
            return ''
        if (len(choices) == 1):
            return choices[0]

        if (not self.exclusive_choices):
            fields = []
            for c in choices:
                f = {"name":c,"displayName":c,"description":"","value":False,"valueType":"boolean"}
                fields.append(f)
            response = ask_user(self, 
                                    {
                                        "title": "Activity Choice",
                                        "description":"Activity Choices ",
                                        "fields": fields
                                    }
                                )
            val=[]
            for c in choices:
                if (response[c]):
                    val.append(c)
            if (not val):
                raise RuntimeError("A choice needs to be made.")
            output = ','.join(val)
        else:
            fields = [{"name":"choices","displayName":"Activity Choice","description":"","value":"Unknown","valueType":"string","valueItems":choices}]
            response = ask_user(self, 
                                    {
                                        "title": "Activity Choice",
                                        "description":"Activity Choices ",
                                        "fields": fields
                                    }
                                )
            output = response["choices"]
            if (not output):
                output = ''
        return output


    def main(self):
        # core = self.core
        # root_node = self.root_node
        # active_node = self.active_node

        # name = core.get_attribute(active_node, 'name')

        # logger.info('ActiveNode at "{0}" has name {1}'.format(core.get_path(active_node), name))

        # core.set_attribute(active_node, 'name', 'newName')

        # commit_info = self.util.save(root_node, self.commit_hash, 'master', 'Python plugin updated the model')
        # logger.info('committed :{0}'.format(commit_info))
        try:
            core = self.core
            active_node = self.active_node
            self.check_if_filled()

            running = True
            activity_definitions  = self.get_all_activity_definitions()
            if (activity_definitions):
                activity_choices = list(activity_definitions.keys())
                user_choice = self.get_user_response(activity_choices)
                if (user_choice and user_choice != "Unknown"):
                    self.get_activity_definition_from_user_choice(activity_definitions,user_choice)
                    self.get_activity_info()
                    self.current_choice = self.get_user_choice_response(self.choices)
                    self.check_activity_info()
                    self.populate_model()
                    name = self.config.get('activity_name', None)
                    if (name):
                        self.core.set_attribute(self.active_node, 'name', name)
                    self.create_message(self.active_node, "Activity Initialized successfully", 'info')
                    commit_info = self.util.save(self.root_node, self.commit_hash, self.branch_name, 'Activity Node Initialized')
                else:
                    self.create_message(self.active_node, "Activity Not Initialized.", 'info')
            else:
                self.create_message(self.active_node, "No activity definitions found in the model/ repository. Activity not initialized", 'info')
            self.result_set_success(True)
            

            # while running:
            #     response = ask_user(self, {"title": "Input is required from BasicInteraction plugin:",
            #     "description":"please fill out the form to your best knowledge",
            #     "fields": [
            #         {
            #             "name":"doFinish",
            #             "displayName": "Do you want to stop plugin execution?",
            #             "description": "If flag is set, the plugin will ask no further questions...",
            #             "value": False,
            #             "valueType": "boolean",
            #             "readOnly": False
            #         },
            #         {
            #         "name": "species",
            #         "displayName": "Animal Species",
            #         "regex": "^[a-zA-Z]+$",
            #         "regexMessage": "Name can only contain English characters!",
            #         "description": "Which species does the animal belong to.",
            #         "value": "Horse",
            #         "valueType": "string",
            #         "readOnly": False
            #         },
            #         {
            #         "name": "age",
            #         "displayName": "Age",
            #         "description": "How old is the animal.",
            #         "value": 3,
            #         "valueType": "number",
            #         "minValue": 0,
            #         "maxValue": 10000,
            #         "readOnly": False,
            #         "writeAccessRequired": True
            #         },
            #         {
            #         "name": "gender",
            #         "displayName": "Gender distribution",
            #         "description": "What is the ratio between females and males?",
            #         "value": 0.5,
            #         "valueType": "number",
            #         "minValue": 0,
            #         "maxValue": 1,
            #         "increment": 0.01
            #         },
            #         {
            #         "name": "carnivore",
            #         "displayName": "Carnivore",
            #         "description": "Does the animal eat other animals?",
            #         "value": False,
            #         "valueType": "boolean",
            #         "readOnly": False
            #         },
            #         {
            #         "name": "isAnimal",
            #         "displayName": "Is Animal",
            #         "description": "Is this animal an animal? [Read-only]",
            #         "value": True,
            #         "valueType": "boolean",
            #         "readOnly": True
            #         },
            #         {
            #         "name": "classification",
            #         "displayName": "Classification",
            #         "description": "",
            #         "value": "Vertebrates",
            #         "valueType": "string",
            #         "valueItems": [
            #             "Vertebrates",
            #             "Invertebrates",
            #             "Unknown"
            #         ]
            #         },
            #         {
            #         "name": "color",
            #         "displayName": "Color",
            #         "description": "The hex color code for the animal.",
            #         "readOnly": False,
            #         "value": "#FF0000",
            #         "regex": "^#([A-Fa-f0-9]{6})$",
            #         "valueType": "string"
            #         },
            #         {
            #         "name": "food",
            #         "displayName": "Food",
            #         "description": "Food preference ordered",
            #         "readOnly": False,
            #         "value": [
            #             "Grass",
            #             "Mushrooms",
            #             "Leaves",
            #             "Antilope",
            #             "Rabbit"
            #         ],
            #         "valueType": "sortable",
            #         "valueItems": [
            #             "Grass",
            #             "Mushrooms",
            #             "Leaves",
            #             "Antilope",
            #             "Rabbit"
            #         ]
            #         },
            #         {
            #         "name": "file",
            #         "displayName": "File",
            #         "description": "",
            #         "value": "",
            #         "valueType": "asset",
            #         "readOnly": False
            #         }
            #     ]})
            #     logger.error(response)
            #     if response != None and response['doFinish']:
            #         running = False


            
            #self.get_definition_from_json()
            #self.get_activity_info()
            #self.check_activity_info()
            #self.populate_model()

            #commit_info = self.util.save(self.root_node, self.commit_hash, self.branch_name, 'Activity Node Initialized')
            

            
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
            self.result_set_error('ActivityInit Plugin: Error encountered.  Check result details.')
            self.result_set_success(False)
            exit()


    # def populate_contexts(self):
    #     num = 0
    #     for name, contents in self.contexts:
    #         num +=1
    #         node = self.create_node(node,name, self.META['ALC.Activity.Context'], POS_X_CONTEXT, num)
    #         self.set_attributes(node,contents)
    #         children = contents.get('parameters',{})
    #         self.populate_child_nodes(node, children,self.META['ALC.Activity.content'])

    # def populate_inputs(self):
    #     num = 0
    #     for name, contents in self.inputs:
    #         num +=1
    #         node = self.create_node(node, name, self.META['ALC.Activity.Input'], POS_X_INPUT, num)
    #         self.set_attributes(node,contents)
            
    # def populate_outputs(self):
    #     num = 0
    #     for name, contents in self.outputs:
    #         num +=1
    #         node = self.create_node(node, name, self.META['ALC.Activity.Output'], POS_X_OUTPUT, num)
    #         self.set_attributes(node,contents)
    #         children = contents.get('attributes',{})
    #         self.populate_child_nodes(node, children,self.META['ALC.Activity.Field'])
    
    # def populate_parameters(self):
    #     num = 0
    #     for name, contents in self.parameters:
    #         num +=1
    #         node = self.create_node(node,name, self.META['ALC.Activity.ParamsTable'], POS_X_PARAMETER, num)
    #         self.set_attributes(node,contents)
    #         children = contents.get('parameters',{})
    #         self.populate_child_nodes(node, children,self.META['ALC.Activity.Parameter'])


