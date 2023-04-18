"""
This is where the implementation of the plugin code goes.
The ALCModelUpdater-class is imported from both run_plugin.py and run_debug.py
"""
import sys
import logging
import json
import os
import re
from webgme_bindings import PluginBase

# Setup a logger
logger = logging.getLogger('ALCModelUpdater')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)  # By default it logs to stderr..
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

#get_member_paths(node, name)
#get_parent(node)[source]
#get_path(node)[source]
#get_pointer_path(node, name)[source]
#load_children(node)[source]
#load_members(node, set_name)[source]
#load_pointer(node, pointer_name)[source]
#get_attribute(node, name)[source]
#set_attribute(node, name, value)[source]
#set_pointer(node, name, target)[source]
#add_member(node, name, member)[source]
#del_member(node, name, path)[source]
#del_pointer(node, name)[source]
#delete_node(node)[source]

class ALCModelUpdater(PluginBase):
    def __init__(self,*args, **kwargs):
        super(ALCModelUpdater, self).__init__(*args, **kwargs)
        self.config = None
        self.training_data_model = None
        self.lec_model = None
        self.eval_data_model = None
        self.assembly_lec_models = {}
        self.task = None
        self.task_dict={}
        self.use_relative_path = False
        self.alc_wkdir = os.getenv('ALC_WORK','')

    def load_task(self):
        logger.info('a')
        self.task = self.config["task"]
        if (self.config.has_key("relative_path")):
            self.use_relative_path = self.config["relative_path"]
            if (self.use_relative_path):
                self.task = os.path.join(self.alc_wkdir,self.task)
        logger.info('b')
        if (not self.task):
            logger.info('self.task is not defined')
            logger.info('c')
            return

        if self.task and os.path.exists(self.task):
            try:
                logger.info('d')
                with open(self.task) as json_file:
                    logger.info('e')
                    self.task_dict = json.load(json_file)
                    logger.info('f')
            except Exception as e:
                logger.info('g')
                logger.info('problem in parsing task/ json file "{0}"'.format(self.task))
                logger.info('exception message "{0}"'.format(e))
                return
        else:
            logger.info('h')
            logger.info('task "{0}" does not exist'.format(self.task))
            return

    def get_task(self):
        self.task = self.config["task"]
        logger.info('task is  "{0}"'.format(self.task))
        try:
            self.task_dict = json.loads(self.task)
            logger.info('parsed task_dict =  {0}'.format(self.task_dict))
        except:
            logger.info('problem in parsing task/ json file "{0}"'.format(self.task))
        return


    def exec_LEC_Setup(self,task_id):
        task_info = self.task_dict[task_id]
        task_name = task_info["name"]
        task_srcs = task_info["src"]
        if (len(task_srcs)==0):
            logger.error('in exec_LEC_Setup: No task sources found')
            return

        task_src = task_info["src"][0]
        if (not task_src):
            logger.error('in exec_LEC_Setup: no source path specified')
            return

        src_node = self.core.load_by_path(self.root_node, task_src)
        if (not src_node):
            logger.error('in exec_LEC_Setup: no source node found in the path {0}'.format(task_src))
            return

        task_dsts = task_info["dst"]
        for task_dst in task_dsts:
            if (not task_dst):
                logger.error('in exec_LEC_Setup: no dst path specified')
                return

            activity_node = self.core.load_by_path(self.root_node, task_dst)
            if (not activity_node):
                logger.error('in exec_LEC_Setup: no activity_node found in the path {0}'.format(task_dst))
                return

            cnodes = self.core.load_children(activity_node)
            lec_node = None
            for child in cnodes:
                if self.core.get_meta_type(child) == self.META["ALCMeta.LEC_Model"]:
                    lec_node = child
                    break
            if (not lec_node):
                logger.error('in exec_LEC_Setup: no lec_node found in the path {0}'.format(task_dst))
                return

            self.core.set_pointer(lec_node,'ModelDataLink',src_node)

        return 0


    def exec_Training_Data_Setup(self,task_id):
        task_info = self.task_dict[task_id]
        task_name = task_info["name"]
        task_srcs = task_info["src"]
        if (len(task_srcs)==0):
            logger.error('in exec_Training_Data_Setup: No task sources found')
            return

        src_nodes = []
        for  n in task_srcs:
            n_node = self.core.load_by_path(self.root_node, n)
            if (not n_node):
                logger.error('in exec_Training_Data_Setup: no source node found in the path {0}'.format(n))
                return
            src_nodes.append(n_node)

        task_dsts = task_info["dst"]
        for task_dst in task_dsts:
            if (not task_dst):
                logger.error('in exec_Training_Data_Setup: no dst path specified')
                continue

            activity_node = self.core.load_by_path(self.root_node, task_dst)
            if (not activity_node):
                logger.error('in exec_Training_Data_Setup: no activity_node found in the path {0}'.format(task_dst))
                continue

            cnodes = self.core.load_children(activity_node)
            training_data_node = None
            for child in cnodes:
                if self.core.get_meta_type(child) == self.META["ALCMeta.TrainingData"]:
                    training_data_node = child
                    break
            if (not training_data_node):
                logger.error('in exec_Training_Data_Setup: no training_data_node found in the path {0}'.format(task_dst))
                continue



        old_vals = self.core.get_member_paths(training_data_node,'Data')
        for o in old_vals:
            self.core.del_member(training_data_node,'Data',o)

        for s in src_nodes:
            self.core.add_member(training_data_node,'Data',s)

        return 0
    

    def exec_Validation_Data_Setup(self,task_id):
        task_info = self.task_dict[task_id]
        task_name = task_info["name"]
        task_srcs = task_info["src"]
        if (len(task_srcs)==0):
            logger.error('in exec_Validation_Data_Setup: No task sources found')
            return

        src_nodes = []
        for  n in task_srcs:
            n_node = self.core.load_by_path(self.root_node, n)
            if (not n_node):
                logger.error('in exec_Validation_Data_Setup: no source node found in the path {0}'.format(n))
                return
            src_nodes.append(n_node)

        task_dsts = task_info["dst"]
        for task_dst in task_dsts:
            if (not task_dst):
                logger.error('in exec_Validation_Data_Setup: no dst path specified')
                continue

            activity_node = self.core.load_by_path(self.root_node, task_dst)
            if (not activity_node):
                logger.error('in exec_Validation_Data_Setup: no activity_node found in the path {0}'.format(task_dst))
                continue

            cnodes = self.core.load_children(activity_node)
            training_data_node = None
            for child in cnodes:
                if self.core.get_meta_type(child) == self.META["ALCMeta.ValidationData"]:
                    training_data_node = child
                    break
            if (not training_data_node):
                logger.error('in exec_Validation_Data_Setup: no validation_data_node found in the path {0}'.format(task_dst))
                continue



        old_vals = self.core.get_member_paths(training_data_node,'Data')
        for o in old_vals:
            self.core.del_member(training_data_node,'Data',o)

        for s in src_nodes:
            self.core.add_member(training_data_node,'Data',s)

        return 0

    
    def exec_Eval_Data_Setup(self,task_id):
        task_info = self.task_dict[task_id]
        task_name = task_info["name"]
        task_srcs = task_info["src"]
        if (len(task_srcs)==0):
            logger.error('in exec_Eval_Data_Setup: No task sources found')
            return

        src_nodes = []
        for  n in task_srcs:
            n_node = self.core.load_by_path(self.root_node, n)
            if (not n_node):
                logger.error('in exec_Eval_Data_Setup: no source node found in the path {0}'.format(n))
                return
            src_nodes.append(n_node)

        task_dsts = task_info["dst"]
        for task_dst in task_dsts:
            if (not task_dst):
                logger.error('in exec_Eval_Data_Setup: no dst path specified')
                continue

            activity_node = self.core.load_by_path(self.root_node, task_dst)
            if (not activity_node):
                logger.error('in exec_Eval_Data_Setup: no activity_node found in the path {0}'.format(task_dst))
                continue

            cnodes = self.core.load_children(activity_node)
            training_data_node = None
            for child in cnodes:
                if self.core.get_meta_type(child) == self.META["ALCMeta.EvalData"]:
                    training_data_node = child
                    break
            if (not training_data_node):
                logger.error('in exec_Eval_Data_Setup: no eval_data_node found in the path {0}'.format(task_dst))
                continue



        old_vals = self.core.get_member_paths(training_data_node,'Data')
        for o in old_vals:
            self.core.del_member(training_data_node,'Data',o)

        for s in src_nodes:
            self.core.add_member(training_data_node,'Data',s)

        return 0


    def exec_Assembly_LEC_Setup(self,task_id):
        task_info = self.task_dict[task_id]
        task_name = task_info["name"]
        task_srcs = task_info["src"]
        if (len(task_srcs)==0):
            logger.error('in exec_Assembly_LEC_Setup: No task sources found')
            return

        task_src = task_info["src"][0]
        if (not task_src):
            logger.error('in exec_Assembly_LEC_Setup: no source path specified')
            return

        src_node = self.core.load_by_path(self.root_node, task_src)
        if (not src_node):
            logger.error('in exec_Assembly_LEC_Setup: no source node found in the path {0}'.format(task_src))
            return

        

        task_dst_lec = task_info["dst_lec"]
        if (not task_dst_lec):
            logger.error('in exec_Assembly_LEC_Setup: no dst_lec path specified ')
            return

        task_dsts = task_info["dst"]
        for task_dst in task_dsts:
            if (not task_dst):
                logger.error('in exec_Assembly_LEC_Setup: no dst path specified')
                continue
            activity_node = self.core.load_by_path(self.root_node, task_dst)
            if (not activity_node):
                logger.error('in exec_Assembly_LEC_Setup: no activity node found in the path {0}'.format(task_dst))
                continue

            cnodes = self.core.load_children(activity_node)
            assembly_node = None
            for child in cnodes:
                if self.core.get_meta_type(child) == self.META["ALCMeta.AssemblyModel"]:
                    assembly_node = child
                    break
            if (not assembly_node):
                logger.error('in exec_Assembly_LEC_Setup: no assembly_node found in the path {0}'.format(task_dst))
                continue

            lec_node = self.core.load_by_path(assembly_node, task_dst_lec)
            if (not lec_node):
                logger.error('in exec_Assembly_LEC_Setup: no lec_node found in the assembly_node in the activity {0}'.format(task_dst))
                continue
            
            self.core.set_pointer(lec_node,'ModelDataLink',src_node)

        return 0


    def exec_Result_Output(self,task_id):
        task_info = self.task_dict[task_id]
        task_name = task_info["name"]
        task_srcs = task_info["src"]
        if (len(task_srcs)==0):
            logger.error('in exec_Result_Output: No task sources found')
            return;

        task_src = task_info["src"][0]
        if (not task_src):
            logger.error('in exec_Result_Output: no source path specified')
            return

        exec_name = ''
        if (task_info.has_key("exec_name")):
            exec_name = task_info["exec_name"]

        return [task_id, self.get_results(task_src,exec_name)]

    def get_results(self,activity_path,exec_name):
        if (not activity_path):
            logger.error('in get_results: No activity_path specified')
            return

        activity_node = self.core.load_by_path(self.root_node, activity_path)
        if (not activity_node):
            logger.error('in get_results: no activity node found in the path {0}'.format(activity_path))
            return

        activity_name = self.core.get_attribute(activity_node,'name')

        cnodes = self.core.load_children(activity_node)
        result_node = None
        for child in cnodes:
            if self.core.get_meta_type(child) == self.META["ALCMeta.Result"]:
                result_node = child
                break
        if (not result_node):
            logger.error('in get_results: no result node found in activity node {0}'.format(activity_name))
            return

        cnodes = self.core.load_children(result_node)
        result_nodes = []
        result_name = 'result-'+exec_name
        for child in cnodes:
            if self.core.get_meta_type(child) != self.META["ALCMeta.DEEPFORGE.pipeline.Data"]:
                continue
            cname = self.core.get_attribute(child,'name')
            if (cname.startswith(exec_name) or cname.startswith(result_name)):
                child_path = self.core.get_path(child)
                ctime = self.core.get_attribute(child,'createdAt')
                cinfo = self.core.get_attribute(child,'datainfo')
                cinfoparsed = cinfo
                if (cinfo == ''):
                    cinfoparsed = {}
                else:
                    try:
                        cinfoparsed = json.loads(cinfo)
                    except:
                        logger.info('problem in parsing data info for {0}'.format(child_path))

                ret = cinfoparsed
                ret.update({
                    "path": child_path,
                    "name": cname,
                    "time": ctime
                })

                result_nodes.append(ret)

        result_nodes.sort(key=lambda x: x.get("time"))
        return result_nodes


    def exec_Update_Status(self,task_id):
        task_info = self.task_dict[task_id]
        task_name = task_info["name"]
        task_srcs = task_info["src"]
        if (len(task_srcs)==0):
            logger.error('in exec_Update_Status: No task sources found')
            return

        task_src = task_info["src"][0]
        if (not task_src):
            logger.error('in exec_Update_Status: no source path specified')
            return

        status_msg = ''
        if (task_info.has_key("status_msg")):
            status_msg = task_info["status_msg"]
        
        if (not status_msg):
            logger.error('Status message is empty')
            return 1
        
        status_node = self.core.load_by_path(self.root_node,task_src)
        cur_status = self.core.get_attribute(status_node,'CurStatus')
        cur_status += '\n '
        cur_status += status_msg
        self.core.set_attribute(status_node,'CurStatus',cur_status)

        return 0

    def isVerificationActivity(self,activity_node):
        logger.info('6')
        ver_types = [self.META["ALCMeta.VerificationSetup"], self.META["ALCMeta.ValidationSetup"],self.META["ALCMeta.SystemIDSetup"]]
        if (self.core.get_meta_type(activity_node) in ver_types):
            logger.info('7')
            return True
        logger.info('8')
        return False


    def exec_Update_Result_Metadata(self,task_id):
        task_info = self.task_dict[task_id]
        task_name = task_info["name"]
        task_dsts = task_info["dst"]
        if (len(task_dsts)==0):
            logger.error('in exec_Update_Result_Metadata: No task destination found')
            return

        task_dst = task_info["dst"][0]
        if (not task_dst):
            logger.error('in exec_Update_Result_Metadata: no destination path specified')
            return

        result_folder = ''
        if (task_info.has_key("result_folder")):
            result_folder = task_info["result_folder"]
            result_folder = os.path.join(self.alc_wkdir,result_folder)
        
        if (not result_folder):
            logger.error('in exec_Update_Result_Metadata: result_folder is not set')
            return 1
        
        if (not (os.path.exists(result_folder) and os.path.isdir(result_folder))):
            logger.error('in exec_Update_Result_Metadata: result_folder {0} does not exist'.format(result_folder))

        
        result_node = self.core.load_by_path(self.root_node,task_dst)
        if (not result_node):
            logger.error('in exec_Update_Result_Metadata: unable to get destination node from path {0}'.format(task_dst))
            return 1
        
        result_meta_data = os.path.join(result_folder,'result_metadata.json')
        if (os.path.exists(result_meta_data)):
            logger.error(' exec_Update_Result_Metadata: result_meta_data.json found in folder {0}'.format(result_folder))
            f = open(result_meta_data,'r')
            result_val = f.read()
            f.close()

            result_name = self.core.get_attribute(result_node,'name')
            result_parent = self.core.get_parent(result_node)
            if (not result_parent):
                logger.error('in exec_Update_Result_Metadata: unable to get result node parent')
                return 1
            activity_node = self.core.get_parent(result_parent)
            if (not activity_node):
                logger.error('in exec_Update_Result_Metadata: unable to get activity node')
                return 1
            activity_name = self.core.get_attribute(activity_node,'name')
            createtime = self.core.get_attribute(result_node, 'createdAt')
            
            result_json = {}
            result_len = -1
            try:
                logger.info('rv '+result_val)
                result_json = json.loads(result_val)
                if (isinstance(result_json,list)):
                    result_len = len(result_json)
                if (result_len==1):
                    result_json = result_json[0]

                logger.info('rjson '+str(result_json))
                if (result_len <= 1):
                    rval = json.dumps(result_json, indent=4, sort_keys=True)
                    logger.info('rval '+rval)
                    self.core.set_attribute(result_node,'datainfo',rval)
                    self.core.set_attribute(result_node,'activity',activity_name)
                    
                    rkeys = result_json.keys()
                    if ("url" in rkeys):
                        logger.info('1')
                        url = result_json["url"]
                        logger.info('2')
                        prefix = 'ipython/notebooks/'
                        if (self.isVerificationActivity(activity_node)):
                            prefix = 'matlab/notebooks/'
                        logger.info('3')
                        url_str = prefix+url
                        nb = self.core.create_child(activity_node, self.META["ALCMeta.Notebook"])
                        logger.info('4')
                        pos={}
                        pos['x']=100
                        pos['y']=100
                        self.core.set_registry(nb, 'position', pos)
                        self.core.set_attribute(nb,'url',url_str)
                        self.core.set_attribute(nb,'name',result_name)
                        logger.info('5')
                else:
                    rval = json.dumps(result_json[0], indent=4, sort_keys=True)
                    self.core.set_attribute(result_node,'datainfo',rval)
                    self.core.set_attribute(result_node,'activity',activity_name)
                    self.core.set_attribute(result_node,'name',result_name+'-0')
                    
                    for i in range(1,result_len):
                        cn = self.core.copy_node(result_node, result_parent)
                        rval = json.dumps(result_json[i], indent=4, sort_keys=True)
                        self.core.set_attribute(cn,'datainfo',rval)
                        self.core.set_attribute(cn,'createdAt',createtime +i)
            except Exception as e:
                    logger.info('failed to parse result_metadata.json in folder : {0}'.format(result_folder))
                    logger.info('exception message "{0}"'.format(e))
                    return 1
        else:
            logger.info('no result metadata found in folder : {0}'.format(result_folder))
        
        slurm_exec_status = os.path.join(result_folder,'slurm_exec_status.txt')
        jobstatus=''
        if (os.path.exists(slurm_exec_status)):
            f = open(slurm_exec_status,'r')
            jobstatus = f.read()
            f.close()
            logger.info('job status read ---:{0}---'.format(jobstatus))
            jobstatus = re.sub(r"[\n\t\s]*", "", jobstatus)
            logger.info('job status after stripping while spaces, new lines ---:{0}---'.format(jobstatus))
            self.core.set_attribute(result_node,'jobstatus', jobstatus)
            logger.info('job status set---:{0}---'.format(jobstatus))
        
        else:
            logger.info('job status is not set')


        return 0
         



    def exec_task(self,task_id):
        task_info = self.task_dict[task_id]
        
        task_name = task_info["name"]

        ret = 1

        logger.info('in exec_task: task_name: "{0}"'.format(task_name))
        
        if (task_name == 'LEC_Setup'):
            return self.exec_LEC_Setup(task_id)

        if (task_name == 'Assembly_LEC_Setup'):
            return self.exec_Assembly_LEC_Setup(task_id)

        if (task_name == 'Training_Data_Setup'):
            return self.exec_Training_Data_Setup(task_id)
        
        if (task_name == 'Eval_Data_Setup'):
            return self.exec_Eval_Data_Setup(task_id)
        
        if (task_name == 'Validation_Data_Setup'):
            return self.exec_Validation_Data_Setup(task_id)
        

        if (task_name == 'Result_Output'):
            return self.exec_Result_Output(task_id)
        
        if (task_name == 'Update_Status'):
            return self.exec_Update_Status(task_id)
        
        if (task_name == 'Update_Result_Metadata'):
            return self.exec_Update_Result_Metadata(task_id)
        

        logger.error('unknown task: {0}'.format(task_name))

        return ret
    
    def exec_tasks(self):
        logger.info('exec_tasks invoked "{0}"'.format(self.task_dict))
        task_ids = self.task_dict.keys()
        ret_vals = {}
        output_filename = ''
        for task_id in task_ids:
            if task_id == "Output_Name":
                output_filename = self.task_dict["Output_Name"]
                continue
            x = self.exec_task(task_id)
            if (task_id == "Result_Output"):
                ret_vals = x[1]
            elif (x and isinstance(x,list)):
                ret_vals[x[0]]=x[1]
        return output_filename, ret_vals

    def output_results(self, filename,vals):
        try:
            outputname = filename
            if (outputname == ''):
                logger.info('no output file specified in task: Result_Output in task file {0}'.format(self.task))
                return

            if (not os.path.exists(os.path.dirname(outputname))):
                logger.info('no directory of the name {0} exists to create output file {1}'.format(os.path.dirname(outputname),outputname))
                return

            with open(outputname, 'w') as outfile:
                json.dump(vals, outfile, indent=4, sort_keys=True)
        except Exception as e:
                logger.info('failed to output json to file : {0}'.format(outputname))
                logger.info('exception message "{0}"'.format(e))
                return
        


    def main(self):
        core = self.core
        root_node = self.root_node
        active_node = self.active_node
        self.config = self.get_current_config()
        #logger.info('{0}'.format(self.META.keys()))
        #self.get_task()
        logger.info('1')
        self.load_task()
        logger.info('2')
        filename, ret  = self.exec_tasks()
        if (ret and (isinstance(ret,dict) or isinstance(ret,list))):
            self.output_results(filename, ret)
        commit_info = self.util.save(root_node, self.commit_hash, 'master', 'Python plugin updated the model')
        logger.info('committed :{0}'.format(commit_info))

