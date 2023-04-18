"""
This is where the implementation of the plugin code goes.
The WorkflowExecutor-class is imported from both run_plugin.py and run_debug.py
"""
import sys
import logging
import json
import os
import re
import time
from pathlib import Path
from webgme_bindings import PluginBase
import traceback

# Setup a logger
logger = logging.getLogger('WorkflowExecutor')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)  # By default it logs to stderr..
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

checkscript = '''
import json
import sys
import os


check_exception_code = 1
incorrect_usage_code = 2
max_iterations = <<max_iteration>>

<<check_script>>


<<update_script>>


def evaluate(iteration_count, result_dict, output_path):

    with open(output_path, 'w') as output_json_file:
        if iteration_count < max_iterations and len(result_dict) > 0:
            first_key = next(iter(result_dict))
            first_dict = result_dict[first_key]

            if isinstance(first_dict, dict) and len(first_dict) > 0:
                second_key = next(iter(first_dict))
                second_dict = first_dict[second_key]

                if isinstance(second_dict, dict) and "Test" in second_dict:
                    test_list = second_dict["Test"]

                    if isinstance(test_list, list) and len(test_list) > 0:
                        test_dict = test_list[0]

                        if isinstance(test_dict, dict) and "info" in test_dict:
                            info_dict = test_dict["info"]
                            if (not terminate(iteration_count,info_dict)):
                                update_dict = update(iteration_count, info_dict)
                                json.dump(update_dict, output_json_file, indent=4, sort_keys=True)

def main(argv):
    iter = 0
    result_dict = {}

    try:
        iteration  = int(argv[0])
    except ValueError:
        #Handle the exception
        print('Expected iteration count as the first argument : ', argv[0])
        sys.exit(incorrect_usage_code)

    if (not os.path.exists(argv[1])):
        print('Expected a valid json file as the second argument : ', argv[1] )
        sys.exit(incorrect_usage_code)
    
    output_path = os.path.dirname(argv[2])
    if (not os.path.isdir(output_path)):
        print('Expected parent directory of third argument to exist: ', argv[2] )
        sys.exit(incorrect_usage_code)

    try:
        with open(argv[1]) as json_file:
            check_dict  = json.load(json_file)
    except ValueError:
        #Handle the exception
        print('Error in parsing json input : ', argv[1] )
        sys.exit(incorrect_usage_code)

    

    try:
        evaluate(iteration, check_dict, argv[2])
    except:
        e = sys.exc_info()[0]
        print('error in executing check: ', e)
        sys.exit(incorrect_usage_code)


    
    

if __name__ == "__main__":
    if (len(sys.argv) < 4):
        print("Usage: {0} iteration input-check-file-path output-update-file-path".format(sys.argv[0]))
        sys.exit(incorrect_usage_code)
    main(sys.argv[1:])
'''



class WorkflowExecutor(PluginBase):
    #def __init__(self,*args, **kwargs):
        #super(WorkflowExecutor, self).__init__(*args, **kwargs)
    def __init__(self, webgme, commit_hash, BRANCH_NAME, ACTIVE_NODE_PATH, ACTIVE_SELECTION_PATHS, NAMESPACE):
        PluginBase.__init__(self, webgme, commit_hash, BRANCH_NAME, ACTIVE_NODE_PATH, ACTIVE_SELECTION_PATHS, NAMESPACE)
        self.webgme = webgme
        self.commit_hash = commit_hash
        self.branch_name = BRANCH_NAME
        self.namespace = NAMESPACE
        
        self.config = None
        self.workflowjson={}
        self.jobs = {}
        self.inputs = {}
        self.outputs = {}
        self.inits = {}
        self.ordered_jobs = []
        self.output_dir = '/alc/workflows/config/'
        self.script_output_dir = '/alc/workflows/scripts/'
        self.workflowname = ''
        self.alcpath = ''
        self.status_path = ''
        self.status_init = []
        self.status_info = {}
        self.status_node = ''
        self.exec_status_node = ''
        self.exec_name = ''
        self.checkScript = {}
        self.updateScript = ''
        self.maxIterations = {}
        self.checkScriptPath = {}
        self.restartJobName = {}
        self.repeat_check_src_job_path = {}
        self.repeat_check_src = {}
        self.workflow_start_jobs = []
        self.start_nodes=[]
        self.property = {}
        self.addedToWorkflowJobs = []
        self.job_json = {}
        self.loop_jobs = {}
        self.loop_jobs_contain_loops = {}
        self.dtval = int(round(time.time() * 1000))
        self.dt = str(self.dtval)
        self.workflow_ouput_dir = ''
        self.prototypes = {}
        self.ep_activity = False
        self.data_stores = {}
        self.data_store_names = []
    
    def meta_(self,metaname):
        metaobj = None
        if (self.META.get("ALCMeta."+metaname)):
            metaobj = self.META["ALCMeta."+metaname]
        if (metaobj):
            return metaobj
        if (self.META.get("ALC_EP_Meta."+metaname)):
            metaobj = self.META["ALC_EP_Meta."+metaname]
        if (metaobj):
            return metaobj

        

    
    def createFolder(self):
        alc_wkdir = os.getenv('ALC_WORK', '')
        logger.info('ALC_WORKING_DIR = ' + alc_wkdir)
        if (not alc_wkdir):
            raise RuntimeError('Environment variable ALC_WORK is unknown or not set')
        if (not os.path.isdir(alc_wkdir)):
            raise RuntimeError('ALC_WORK: ' + alc_wkdir + ' does not exist')
        jupyter_dir = os.path.join(alc_wkdir, 'jupyter')
        logger.info('jupyter_dir = ' + jupyter_dir)
        if (not os.path.isdir(jupyter_dir)):
            raise RuntimeError('jupyter directory : ' + jupyter_dir + ' does not exist in ALC_WORK')

        projInfo = self.project.get_project_info()
        pname = projInfo['_id']
        modelName = self.core.get_attribute(self.active_node, 'name')
        projectName = pname.replace('+', '_')
        
        self.workflow_ouput_dir = os.path.join(jupyter_dir, 'Workflows', projectName, modelName,self.dt)
        if not os.path.isdir(self.workflow_ouput_dir):
            os.makedirs(self.workflow_ouput_dir)

    
    def createWorkflowJson(self):
        projInfo = self.project.get_project_info()
        self.workflowjson['projectName'] =  projInfo['name']
        self.workflowjson['owner'] =  projInfo['owner']
        self.workflowjson['genericActiveNode']= self.alcpath #'/y'
        self.workflowjson['buildRoot'] = '/alc/workflows/automate/'+self.workflowname
        self.workflowjson['workflowName']=self.workflowname
        #self.workflowjson['checkScriptPath'] =  ''
       
        self.workflowjson['jobs'] = [ ]
        self.workflowjson['statusPath']=''
        self.workflowjson['maxIterations']= self.maxIterations
        #self.workflowjson['launchExpt'] = 0
        #self.workflowjson['launchActivity'] = 0

        self.workflowjson['datastores']= []
        for k in self.data_stores:
            self.workflowjson['datastores'].append(self.addDataStoreToWorkflow(k))



        self.workflowjson['start_jobs']=[]
        for k in self.workflow_start_jobs:
            self.workflowjson['start_jobs'].append(self.jobs[k]['name'])#[-1])


        #self.workflowjson['repeat_check']= 0
        #if (self.repeat_check_path):
            #self.workflowjson['repeat_check']=1
        #    self.checkScriptPath = os.path.join(self.output_dir,self.workflowname+'_script','check.py')
        #else:
            #self.repeat_check_src_job_path=''
            #self.repeat_check_dst_job_path=''
        
        

        self.loop_jobs = {}
        self.loop_jobs_contain_loops = {}
        for j in self.ordered_jobs:
            loop_parent = self.jobs[j]['loop_parent']
            if (loop_parent == ''):
                continue
            if (loop_parent not in self.loop_jobs.keys()):
                self.loop_jobs[loop_parent] = []
                self.loop_jobs_contain_loops[loop_parent] = []
            if (j not in self.loop_jobs[loop_parent]):
                self.loop_jobs[loop_parent].append(j)
                if (self.jobs[j]['job_type']=='Loop'):
                    self.loop_jobs_contain_loops[loop_parent].append(j)
        

        for j in self.ordered_jobs:
            self.status_info[j] = self.updateStatusInit(j)

            if (j in self.addedToWorkflowJobs):
                continue

            jobinfo = self.addJobToWorkflow(j)
            self.job_json[j] = jobinfo
            
            self.workflowjson['jobs'].append(jobinfo)
            self.addedToWorkflowJobs.append(j)

                
        
        #if (self.repeat_check_path):
        #    ret = self.addCheckToWorkflow(j)
        #    self.workflowjson['jobs'].append(ret)
        if (self.ep_activity):
            self.workflowjson['prototypes']=self.prototypes
            #self.workflowjson['launchActivity'] =1
        #else:
        #    self.workflowjson['launchExpt'] =1
        
        self.createAndUpdateStatusExec()
    
    def getExecName(self):
        cname = self.config['Name']
        #FIXME: check for uniquness before setting exec-name
        self.exec_name = cname
       



    def createAndUpdateStatusExec(self):
        if (not self.status_node):
            self.createStatusNode()
        self.exec_status_node = self.core.create_child(self.status_node,self.meta_('WFExecStatus'))
        self.getExecName()
        self.core.set_attribute(self.exec_status_node, 'name', self.exec_name)
        status_init = '\n'.join(self.status_init)
        self.core.set_attribute(self.exec_status_node, 'Init', status_init)
        status_info = json.dumps(self.status_info)
        self.core.set_attribute(self.exec_status_node, 'Info', status_info)
        exec_status_path = self.core.get_path(self.exec_status_node)
        self.workflowjson['statusPath']=exec_status_path


        
    

    def createStatusNode(self):
        self.status_node = self.core.create_child(self.active_node,self.meta_('Status'))




    
    def updateStatusInit(self, j):
        ret = {}
        job = self.jobs[j]
        job_name = job['name']
        self.status_init.append('Job --- '+job_name[-1])
        activities = job['activities']
        akeys = activities.keys()
        
        for a in akeys:
            aname = activities[a]
            self.status_init.append('   Acitivity --- '+aname)
            ret[aname]= a
        return ret


    def addDataStoreToWorkflow(self, ds):
        datastore = self.data_stores[ds]
        ret = {}
        ret['job_name'] = datastore['name']
        if datastore['job_ref']:
            ret['job'] = { 'workflow':datastore['job_ref'][0], 'job':datastore['job_ref'][1]}
        if datastore['data']:
            ret['data']=[]
            for d in datastore['data']:
                dnode = self.core.load_by_path(self.root_node, d)
                if dnode:
                    resultstr = self.core.get_attribute(dnode,'datainfo')
                    if resultstr:
                        try:
                            result_metadata  = json.loads(resultstr)
                            d_val={'path':d, 'result_metadata':result_metadata}
                            ret['data'].append(d_val)
                        except:
                            #Handle the exception
                            print('Error in result metadata json: ', resultstr )
        
        return ret

    def addJobToWorkflow(self, j):
        job = self.jobs[j]
        ret = {}
        ret['job_name']= job['name'][-1]
        print ('in job ',ret['job_name'] )
        ret['prev_jobs']= []
        ret['activities_map'] = {}
        ret['inits']= []
        ret['inputs'] = {}
        #ret['update'] = 0
        ret['check'] = 0
        ret['job_type']= job['job_type']
        job_type = job['job_type']
        ret['job_subtype']= job['job_subtype']
        ret['loop_parent']=''
        ret['loop_parent_path']=''
        if (job['loop_parent']):
            ret['loop_parent']= self.jobs[job['loop_parent']]['name']
            ret['loop_parent_path']=job['loop_parent']

        ret['next_job_paths']=job['next_jobs']
        ret['next_jobs']=[]
        job_node  = self.core.load_by_path(self.root_node, j)
        job_parent = self.core.get_parent(job_node)
        for k in job['next_jobs']:
            output_node = self.core.load_by_path(self.root_node, k)
            output_parent = self.core.get_parent(output_node)
            if (output_parent != job_parent):
                continue
            name = self.jobs[k]['name']
            ret['next_jobs'].append(name[-1])
        

        ret['next_branch_true']=[]
        ret['next_branch_true_paths']=job['branch_true']
        for k in job['branch_true']:
            name = self.jobs[k]['name']
            ret['next_branch_true'].append(name[-1])


        ret['next_branch_false']=[]
        ret['next_branch_false_paths']=job['branch_false']
        for k in job['branch_false']:
            name = self.jobs[k]['name']
            ret['next_branch_false'].append(name[-1])


        ret["script"]= '\''+job['script']+'\''
        ret["loop_type"]=job["loop_type"]
        ret["loop_start_path"]=job["loop_start"]
        ret["loop_start"]=[]

        for k in job["loop_start"]:
            name = self.jobs[k]['name']#[-1]
            ret['loop_start'].append(name)
        
        ret["loop_iter_vars"] = re.sub('\\n',',',job["loop_iter_vars"])
        ret["loop_vars"] = job["loop_vars"]

        ret["loop_jobs"]=[]

        



        #address input_vars, output_vars, loop_vars, loop_type, loop_iter_vars, loop_start
        #self.jobs[job_path]["input_vars"]=[]
        #self.jobs[job_path]["output_vars"]=[]
        #self.jobs[job_path]["loop_vars"]={}
        #self.jobs[job_path]["loop_type"]=''
        #self.jobs[job_path]["loop_iter_vars"]=''
        #self.jobs[job_path]["loop_start"]=[]




        #if (self.repeat_check_dst_job_path == j):
        #    ret['update'] = 1
        
        if (j in self.repeat_check_src_job_path.keys()):
            repeat_check_path = self.repeat_check_src_job_path[j]
            ret['check'] = 1
            ret['check_script_path'] = self.checkScriptPath[repeat_check_path]
            ret['max_iterations'] = self.maxIterations[repeat_check_path]
            ret['restart_at_job'] = self.restartJobName[repeat_check_path]

        job_node  = self.core.load_by_path(self.root_node, j)
        job_parent = self.core.get_parent(job_node)
        #k=-1
        for k in job['input_parent_jobs']:
            #k +=1
            input_node = self.core.load_by_path(self.root_node, k)
            #input_parent = self.core.get_parent(input_node)
            #if (input_parent == job_parent):
            #    continue
            if (self.core.get_meta_type(input_node)== self.meta_("DataStore")):
                continue
            name = self.jobs[k]['name']
            ret['prev_jobs'].append(name[-1])
            
        
        
        activities = job['activities']
        akeys = activities.keys()
        anames = []
        
        for a in akeys:
            aname = activities[a]
            ret['activities_map'][aname]=a
            a_node = self.core.load_by_path(self.root_node, a)
            if (job_type == 'LaunchActivity'):
                self.prototypes[aname]=a
            ret['activities_name']= aname
            anames.append(aname)
        
        if (job_type == 'LaunchExperiment'):
            ret['inputs'] = []
            inputs = job['inputs']
            for i in inputs:
                input_job = self.inputs[i]
                input_val = {}
                input_val['task_name']= input_job['name']
                input_val['input_job']=''
                if (len(input_job['src']) > 0 ):
                    prev_job_path = input_job['src'][0]
                    prev_job_node = self.core.load_by_path(self.root_node, prev_job_path)
                    prev_job_name = self.core.get_attribute(prev_job_node,'name')
                    input_val['input_job']= []
                    input_val['port'] = ''
                    logger.info(' input_job_name "{0}" '.format(prev_job_name))
                    logger.info(' input_job_meta type "{0}" '.format(self.core.get_meta_type(prev_job_node)))
                    logger.info(' datastore_meta type "{0}" '.format(self.meta_("DataStore")))
                    if (self.core.get_meta_type(prev_job_node)== self.meta_("DataStore")):
                        input_val['input_job'].append(prev_job_name)
                    else:

                        input_job_parent_len = len(self.jobs[prev_job_path]['name']) 
                        for idx_v in range(0,input_job_parent_len):
                            pn = self.jobs[prev_job_path]['name'][idx_v]
                            pv = self.jobs[prev_job_path]['parent_loop_num'][idx_v]
                            if (pv==''):
                                input_val['input_job'].append(pn)
                            else:
                                input_val['input_job'].append([pn,pv])
                        if (len(input_job['port'])):
                            input_val['port'] = input_job['port'][0]
                    input_val['in_loop']=input_job['inside_loop']
                task = input_job['task']
                if (task == 'TrainingData'):
                    input_val['name'] = 'Training_Data_Setup'
                if (task == 'EvalData'):
                    input_val['name'] = 'Eval_Data_Setup'
                if (task == 'ValidationData'):
                    input_val['name'] = 'Validation_Data_Setup'
                if (task == 'Training_LEC'):
                    input_val['name'] = 'LEC_Setup'
                if (task == 'Assembly_LEC'):
                    input_val['name'] = 'Assembly_LEC_Setup'
                input_val['dst'] = list(akeys)
                input_val['dst_lec']=''
                if (task == 'Assembly_LEC'):
                    input_val['dst_lec']=input_job['dstLEC']
                input_val['loop'] = input_job['loop']
                
                ret['inputs'].append(input_val)
        else:
            print('adding inputs for activity or branch or transform')
            ret['inputs']={}
            inputs = job['inputs']
            for i in inputs:
                input_job = self.inputs[i]
                input_val = {}
                input_val['task_name']= input_job['name']
                input_val['input_job']=[]
                print(input_job['name'])
                #if (len(input_job['src']) > 0 ):
                for prev_job_path in input_job['src']:
                    #prev_job_path = input_job['src'][0]
                    prev_job_node = self.core.load_by_path(self.root_node, prev_job_path)
                    prev_job_name = self.core.get_attribute(prev_job_node,'name')
                    #input_val['input_job']= []
                    input_info = [] 
                    if (self.core.get_meta_type(prev_job_node)== self.meta_("DataStore")):
                        input_info.append(prev_job_name)
                    else:
                        input_job_parent_len = len(self.jobs[prev_job_path]['name'])
                        
                        for idx_v in range(0,input_job_parent_len):
                            pn = self.jobs[prev_job_path]['name'][idx_v]
                            pv = self.jobs[prev_job_path]['parent_loop_num'][idx_v]
                            print ('pn= ',str(pn))
                            print ('pv= ',str(pv))
                            if (pv==''):
                                input_info.append(pn)
                            else:
                                input_info.append([pn,pv])
                    input_val['input_job'].append(input_info)
                    input_val['port'] = ''
                    if (len(input_job['port'])):
                        input_val['port'] = input_job['port'][0]
                    input_val['in_loop']=input_job['inside_loop']
                print('handling input_name')
                print(str(input_val['input_job']))

                if ('input_name' not in input_job) or len(input_job['input_name'])==0:
                    ret['inputs'][input_val['task_name']]=input_val['input_job']
                else:
                    print('for each input_name')
                    for iname in input_job['input_name']:
                        print (iname)
                        ret['inputs'][iname]=input_val['input_job']
                
        inits = job['inits']
        for i in inits:
            init_job = self.inits[i]
            init_val = {}
            #init_val['task_name']= init_job['name']
            init_val['src']=[]
            if (len(init_job['src']) > 0 ):
                init_val['src']=init_job['src']
            task = init_job['task']
            if (task == 'TrainingData'):
                init_val['name'] = 'Training_Data_Setup'
            if (task == 'EvalData'):
                init_val['name'] = 'Eval_Data_Setup'
            if (task == 'ValidationData'):
                init_val['name'] = 'Validation_Data_Setup'
            if (task == 'Training_LEC'):
                init_val['name'] = 'LEC_Setup'
            if (task == 'Assembly_LEC'):
                init_val['name'] = 'Assembly_LEC_Setup'
            init_val['dst'] = list(akeys)
            init_val['dst_lec']=''
            if (task == 'Assembly_LEC'):
                init_val['dst_lec']=init_job['dstLEC']
            init_val['loop'] = init_job['loop']

            ret['inits'].append(init_val)


        if (job_type =='Loop'):
            child_jobs =  self.loop_jobs[j]
            for cj in child_jobs:
                if (cj in self.addedToWorkflowJobs):
                    continue
                job_info = self.addJobToWorkflow(cj)
                ret['loop_jobs'].append(job_info)
                self.addedToWorkflowJobs.append(cj)
        
        ret["use_parameters"]=[]
        if (len(job["property"])):
            for p in job["property"]:
                if (self.property[p] and self.property[p]['src']):
                    for param in self.property[p]['src']:
                        ret["use_parameters"].append(param)

        
        ret['previous_jobs']= ret['prev_jobs']
        ret['next_jobs_true']=ret['next_jobs']
        ret['next_jobs_false'] = ret['next_branch_false']
        if ret['next_branch_true']:
            ret['next_jobs_true']=ret['next_branch_true']
        #if self.prototypes and ret['job_type']!='LaunchExperiment':
        del ret['next_jobs']
        del ret['prev_jobs']
        del ret['next_branch_true']
        del ret['next_branch_false']
        del ret['next_branch_false_paths'] 
        del ret['next_branch_true_paths']
        del ret['next_job_paths']

        return ret
    
    #def addCheckToWorkflow(self, j):
    #    job = self.jobs[j]
    #    ret = {}
    #    ret['job_name']= 'repeat_check'
    #    ret['prev_jobs']= [self.jobs[self.repeat_check_src_job_path]['name']]
    #    ret['activities_map'] = {}
    #    ret['inits']= []
    #    ret['inputs'] = []
    #    ret['update'] = 0
    #    ret['check'] = 1
    #    return ret
 

    def orderJobs(self):
        jkeys = self.jobs.keys()
        while True:
            count =0
            for j in jkeys:
                logger.info(' order job:  looking at job "{0}" '.format(self.jobs[j]['name']))
                if (j not in self.ordered_jobs and self.checkAllParentsAdded(j)):
                    logger.info(' addking to ordered job "{0}" '.format(self.jobs[j]['name']))
                    self.ordered_jobs.append(j)
                    count +=1
            if ((count == 0) or (len(self.ordered_jobs) == len(jkeys))):
                logger.info(' order job count : "{0}" '.format(count))
                break
        
        #self.repeat_check_src_job_path = self.ordered_jobs[-1]
        #self.repeat_check_dst_job_path = self.ordered_jobs[0]
    
    def checkAllParentsAdded(self,j):
        job = self.jobs[j]
        prev_jobs = job['input_jobs']
        loop_parent = job['loop_parent']
        if (j in self.workflow_start_jobs):
            logger.info(' in start jobs')
            return 1
        dskeys = self.data_stores.keys()
        if (j in dskeys):
            return 1
        
        if ((len(prev_jobs) == 0) and (loop_parent=='')) :
            logger.info(' no prev job, no loop parent')
            return 1
        
        if (loop_parent and (loop_parent in self.ordered_jobs)) :
            if (j in self.jobs[loop_parent]['loop_start']):
                logger.info(' loop parent, in loop_start')
                return 1
            if (len(prev_jobs) == 0) :
                logger.info(' no pev job, loop_parent in ordered jobs')
                return 1
        elif loop_parent:
            return 0

        

        for p in prev_jobs:
            if (j in dskeys):
                continue
            if (p not in self.ordered_jobs):
                return 0
        logger.info(' default ')
        return 1


    def getJobName(self, job):
        ret = []
        num = []
        name = self.core.get_attribute(job,'name')
        ret.append(name)
        num.append('')
        logger.info('in get job '+name)
        parent  = self.core.get_parent(job)
        loop_parent_count = 0
        while True:
            if (self.core.get_meta_type(parent) == self.meta_("Workflow")):
                break
            name = self.core.get_attribute(parent,'name')
            ret.append(name)
            numstr = ''
            if (self.core.get_meta_type(parent) == self.meta_("Loop")):
                loop_parent_count +=1
                if (loop_parent_count > 1):
                    numstr = 'max'
            num.append(numstr)
            parent = self.core.get_parent(parent)

        logger.info('in get job 0 '+str(ret))    
        ret.reverse()
        num.reverse()
        logger.info('in get job 1 '+str(ret))
        return ret,num

    def get_workflow_job_path(self, job):
        ret = []
        name = self.core.get_attribute(job,'name')
        ret.append(name)
        parent  = self.core.get_parent(job)
        workflowname = ''
        while True:
            if (self.core.get_meta_type(parent) == self.meta_("Workflow")):
                workflowname = self.core.get_attribute(parent,'name')
                break
            name = self.core.get_attribute(parent,'name')
            ret.append(name)
            parent = self.core.get_parent(parent)

        ret.reverse()
        return workflowname, ret
        



    def addJob(self, job):
        job_path = self.core.get_path(job)
        keys = self.jobs.keys()
        name, parent_loop_num = self.getJobName(job) #self.core.get_attribute(job,'name')
        ret = False
        if job_path not in keys:
            self.jobs[job_path]={}
            self.jobs[job_path]['name']=name
            self.jobs[job_path]['parent_loop_num']=parent_loop_num
            self.jobs[job_path]['activities']={}
            self.jobs[job_path]['input_jobs']=[]
            self.jobs[job_path]['input_parent_jobs']=[]
            self.jobs[job_path]['loop_parent']=''
            self.jobs[job_path]['next_jobs']=[]
            self.jobs[job_path]['branch_true']=[]
            self.jobs[job_path]['branch_false']=[]
            self.jobs[job_path]['inputs']=[]
            self.jobs[job_path]['inits']=[]
            self.jobs[job_path]['job_type']="ALC_Job"
            self.jobs[job_path]['job_subtype']=""
            self.jobs[job_path]["input_vars"]=[]
            self.jobs[job_path]["script"]=''
            self.jobs[job_path]["output_vars"]=[]
            self.jobs[job_path]["property"]=[]
            self.jobs[job_path]["loop_vars"]={}
            self.jobs[job_path]["loop_type"]=''
            self.jobs[job_path]["loop_iter_vars"]=''
            self.jobs[job_path]["loop_start"]=[]
        
        parent = self.core.get_parent(job)
        if (self.core.get_meta_type(parent) == self.meta_("Loop")):
            self.jobs[job_path]['loop_parent'] = self.core.get_path(parent)
        
        if (self.core.get_meta_type(job) == self.meta_("Transform")):
            self.jobs[job_path]["job_type"]="Transform"
            self.jobs[job_path]["job_subtype"]=self.core.get_attribute(job,"Transform")
            self.jobs[job_path]["script"]=self.core.get_attribute(job,"script")
        
        if (self.core.get_meta_type(job) == self.meta_("Branch")):
            self.jobs[job_path]["job_type"]="Branch"
            self.jobs[job_path]["script"]=self.core.get_attribute(job,"script")
            
        
        if (self.core.get_meta_type(job) == self.meta_("Loop")):
            ret = True
            self.jobs[job_path]["job_type"]="Loop"
            self.jobs[job_path]["script"]=self.core.get_attribute(job,"script")
            self.jobs[job_path]["loop_type"]=self.core.get_attribute(job,"Loop_Type")
            self.jobs[job_path]["loop_iter_vars"]=self.core.get_attribute(job,"Loop_Iter_Vars")
        
        cnodes = self.core.load_children(job)
        for child in cnodes:
            if self.core.get_meta_type(child) == self.meta_("WF_Input"):
                cname = self.core.get_attribute(child,'name')
                cpath = self.core.get_path(child)
                self.jobs[job_path]["input_vars"].append(cpath)
            
            if self.core.get_meta_type(child) == self.meta_("WF_Output"):
                cname = self.core.get_attribute(child,'name')
                cpath = self.core.get_path(child)
                self.jobs[job_path]["output_vars"].append(cpath)
            
            if self.core.get_meta_type(child) == self.meta_("Loop_Var"):
                cname = self.core.get_attribute(child,'name')
                values = self.core.get_attribute(child,'Values')
                self.jobs[job_path]["loop_vars"][cname]=values

            if self.core.get_meta_type(child) == self.meta_("WF_Start"):
                connssrc = self.core.get_collection_paths(child,'src')
                for cs in connssrc:
                    conn = self.core.load_by_path(self.root_node, cs)
                    npath = self.core.get_pointer_path(conn,'dst')
                    node = self.core.load_by_path(self.root_node,npath)
                    nodename = self.core.get_attribute(node,'name')
                    self.jobs[job_path]["loop_start"].append(npath)
        
        return ret

  
    
    def addActivities(self,job):
        job_path = self.core.get_path(job)
        anodes = self.core.load_children(job)
        for child in anodes:
            if self.core.get_meta_type(child) == self.meta_("Activities"):
                activity_nodes = self.core.load_members(child, 'JData')
                for actnode in activity_nodes:
                    name = self.core.get_attribute(actnode,'name')
                    path = self.core.get_path(actnode)
                    self.jobs[job_path]['activities'][path]=name
                    self.jobs[job_path]['job_type']='LaunchExperiment'
            if self.core.get_meta_type(child) == self.meta_("Activity"):
                self.ep_activity = True
                activity_node = child
                actnode = self.core.get_base(activity_node)
                name = self.core.get_attribute(actnode, 'name')
                path = self.core.get_path(actnode)
                self.jobs[job_path]['activities'][path]=name
                self.jobs[job_path]['job_type']='LaunchActivity'
                

    
    def addJobDependency(self,src_job, dst_job, is_in_loop,branch,actual_src_job):
        src_path = self.core.get_path(src_job)
        dst_path = self.core.get_path(dst_job)
        src_name = self.core.get_attribute(src_job,'name')
        dst_name = self.core.get_attribute(dst_job,'name')
        orig_src_path = src_path

        logger.info('++++++++++++++ entring  job dependency '+ src_name + ' - > '+ dst_name + '----------'+branch)
        
        is_data_store_src = False

        if (self.core.get_meta_type(src_job)== self.meta_("DataStore")):
            is_data_store_src = True
            logger.info('++++++++++++++ data store')
            


        #ignore cases when a loop's child job outputs to loop.
        if (not is_data_store_src) and (self.jobs[src_path]['loop_parent'] and self.jobs[src_path]['loop_parent'] == dst_path):
            return

        if (is_in_loop):
            self.jobs[dst_path]['loop_parent']=src_path
            
        if (actual_src_job):
            src_path = self.core.get_path(actual_src_job)
            src_name = self.core.get_attribute(actual_src_job,'name')
        

        if (src_path not in self.jobs[dst_path]['input_jobs']):
            #if (not is_in_loop):
            self.jobs[dst_path]['input_jobs'].append(src_path)
            if (not is_in_loop):
                if (not actual_src_job):
                    self.jobs[dst_path]['input_parent_jobs'].append(src_path)
                else:
                    self.jobs[dst_path]['input_parent_jobs'].append(orig_src_path)

            #if (is_in_loop):
            #    self.jobs[dst_path]['loop_parent']=src_path

            logger.info('added input job dependency '+ src_name + ' - > '+ dst_name)

        if (not is_data_store_src) and ((not is_in_loop) and (dst_path not in self.jobs[orig_src_path]['next_jobs'])):
            self.jobs[orig_src_path]['next_jobs'].append(dst_path)
            if (branch == 'TRUE'):
                self.jobs[orig_src_path]['branch_true'].append(dst_path)
            if (branch == 'FALSE'):
                self.jobs[orig_src_path]['branch_false'].append(dst_path)

            
            logger.info('added next job dependency '+ src_name + ' - > '+ dst_name)

    def addInputs(self,job):
        job_path = self.core.get_path(job)
        inodes = self.core.load_children(job)
        for child in inodes:
            if ((self.core.get_meta_type(child) == self.meta_("Jobs_Input")) or (self.core.get_meta_type(child) == self.meta_("WF_Input"))):
                path = self.core.get_path(child)
                is_generic_input = False
                if (self.core.get_meta_type(child) == self.meta_("WF_Input")):
                    is_generic_input = True
                self.inputs[path]={'name':'','task':'', 'input_name':[],'src':[], 'dstLEC':'', 'loop': '', 'generic_input': is_generic_input, 'port':[], 'inside_loop':False}
                self.jobs[job_path]['inputs'].append(path)

    def addOutputs(self,job):
        job_path = self.core.get_path(job)
        inodes = self.core.load_children(job)
        for child in inodes:
            if (self.core.get_meta_type(child) == self.meta_("WF_Output")):
                path = self.core.get_path(child)
                self.outputs[path]={'name':'', 'src':[],'port':[]}
                if (path not in self.jobs[job_path]['output_vars']):
                    self.jobs[job_path]['output_vars'].append(path)
    
    def addProperties(self,job):
        job_path = self.core.get_path(job)
        inodes = self.core.load_children(job)
        for child in inodes:
            if self.core.get_meta_type(child) != self.meta_("WF_Property"):
                continue
            path = self.core.get_path(child)
            self.property[path]={'name':'', 'src':[]}
            if (path not in self.jobs[job_path]['property']):
                self.jobs[job_path]['property'].append(path)
            self.property[path]['name']= self.core.get_attribute(child,'name')
            connsdst = self.core.get_collection_paths(child,'dst')
            for c in connsdst:
                conn = self.core.load_by_path(self.root_node, c)
                if self.core.get_meta_type(conn) == self.meta_("WFExecutionSeq"): 
                    npath = self.core.get_pointer_path(conn,'src')
                    node = self.core.load_by_path(self.root_node,npath)
                    nodename = self.core.get_attribute(node,'name')
                    if self.core.get_meta_type(node) == self.meta_("Loop_Var"):
                        self.property[path]['src'].append(nodename)

        
    def addInits(self,job):
        job_path = self.core.get_path(job)
        inodes = self.core.load_children(job)
        for child in inodes:
            if self.core.get_meta_type(child) != self.meta_("Init_Value"):
                continue
            path = self.core.get_path(child)
            self.inits[path]={'name':'','task':'', 'src':[], 'dstLEC':'', 'loop': ''}
            self.jobs[job_path]['inits'].append(path)
    
    def isLECTrainingJob(self,job):
        ret = False
        inodes = self.core.load_children(job)
        for child in inodes:
            if self.core.get_meta_type(child) == self.meta_("Jobs_Output"):
                outtype = self.core.get_attribute(child,'OutType')
                if (outtype == 'LEC'):
                    return True
        return ret


    def processInputs(self,job):
        job_path = self.core.get_path(job)
        inodes = self.core.load_children(job)
        for child in inodes:
            if ((self.core.get_meta_type(child) != self.meta_("Jobs_Input")) and (self.core.get_meta_type(child) != self.meta_("WF_Input"))):
                continue
            path = self.core.get_path(child)
            self.inputs[path]['name'] = self.core.get_attribute(child,'name')
            self.inputs[path]['input_name'] = []

            connssrc = self.core.get_collection_paths(child,'src')
            for c in connssrc:
                conn = self.core.load_by_path(self.root_node, c)
                if self.core.get_meta_type(conn) != self.meta_("In_Val"):
                    continue
                npath = self.core.get_pointer_path(conn,'dst')
                node = self.core.load_by_path(self.root_node,npath)
                if self.core.get_meta_type(node) != self.meta_("Input"):
                    continue
                nodename = self.core.get_attribute(node,'name')
                self.inputs[path]['input_name'].append(nodename)
            if self.core.get_meta_type(child) != self.meta_("Jobs_Input"):
                continue
            self.inputs[path]['task'] = self.core.get_attribute(child,'Set')
            inptype = self.core.get_attribute(child,'InpType')
            if (inptype == 'Data'):
                looptype = self.core.get_attribute(child,'Loop_Setting')
                if (looptype == 'USE_ALL'):
                    self.inputs[path]['loop'] = 'all'
            if (self.inputs[path]['task'] in  ['Assembly_LEC']):
                dstlecname =self.core.get_attribute(child,'Target_Assembly_LEC')
                assembly_lec_path = self.getAssemblyLECPath(dstlecname, job_path)
                self.inputs[path]['dstLEC'] = assembly_lec_path
    
    def getAssemblyLECPath(self,lecname, job_path):
        logger.info('getAssemblyLECPath "{0}" '.format(lecname))
        actpaths  = self.jobs[job_path]['activities'].keys()
        if (not actpaths):
            return ''
        act_path = list(actpaths)[0]
        act_node  = self.core.load_by_path(self.root_node,act_path)
        cnodes = self.core.load_children(act_node)
        for child in cnodes:
            if self.core.get_meta_type(child) != self.meta_("AssemblyModel"):
                continue
            logger.info('getLECPathFromAssembly ')
            return self.getLECPathFromAssembly(child,lecname)
    
    def getLECPathFromAssembly(self,assembly, lecname):
        st = self.core.load_sub_tree(assembly)
        for node in st:
            if self.core.get_meta_type(node) != self.meta_("LEC_Model"):
                continue
            name = self.core.get_attribute(node,'name')
            logger.info('getLECPathFromAssembly name "{0}" '.format(name))
            if (name == lecname):
                lecpath = self.core.get_path(node)
                assemblypath = self.core.get_path(assembly)
                logger.info('lec path "{0}" assembly path "{1}" '.format(lecpath,assemblypath))
                ret = lecpath.replace(assemblypath,'')
                return ret
        logger.info('getLECPathFromAssembly  null')
        return ''

    
    def processInits(self,job):
        job_path = self.core.get_path(job)
        isLECTraining = self.isLECTrainingJob(job)
        inodes = self.core.load_children(job)
        for child in inodes:
            if self.core.get_meta_type(child) != self.meta_("Init_Value"):
                continue
            path = self.core.get_path(child)
            self.inits[path]['name'] = self.core.get_attribute(child,'name')
            self.inits[path]['task'] = self.core.get_attribute(child,'Set')
            if (isLECTraining):
                loop_setting = self.core.get_attribute(child,'Loop_Setting')
                if (loop_setting == 'USE_LATEST'):
                    self.inits[path]['loop'] = 'latest'
            if (self.inits[path]['task'] in  ['Assembly_LEC', 'Training_LEC', 'PlantModel']):
                lecref = self.core.get_pointer_path(child,'ModelDataLink')
                if (lecref):
                    self.inits[path]['src'] = [lecref]
                
            if (self.inits[path]['task'] in  ['TrainingData', 'EvalData']):
                datasets = self.core.get_member_paths(child,'Init_Data')
                if (datasets and len(datasets)>0):
                    self.inits[path]['src'] = datasets
            if (self.inits[path]['task'] in  ['Assembly_LEC']):
                dstlecname =self.core.get_attribute(child,'TargetAssembly')
                assembly_lec_path = self.getAssemblyLECPath(dstlecname, job_path)
                self.inits[path]['dstLEC'] = assembly_lec_path


    def addInputSrc(self, input_node,job_node,port_name,is_in_loop,actual_src_job):
        input_path = self.core.get_path(input_node)
        if (not job_node):
            logger.info('no job node')
        else:
            logger.info('job node ')
            logger.info('job node  info "{0}" '.format(job_node))
        
        job_path = self.core.get_path(job_node)
        if (actual_src_job):
            job_path = self.core.get_path(actual_src_job)
            logger.info('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ had actual_src_job')
        if (job_path not in self.inputs[input_path]['src']):
            self.inputs[input_path]['src'].append(job_path)
            #if (port_name):
            #    self.inputs[input_path]['port'].append(port_name)
            #    if (is_in_loop):
            #        self.inputs[input_path]['inside_loop'] = True
    
    def addOutputSrc(self, input_node,job_node,port_name):
        input_path = self.core.get_path(input_node)
        job_path = self.core.get_path(job_node)
        if (job_path not in self.outputs[input_path]['src']):
            self.outputs[input_path]['src'].append(job_path)
            if (port_name):
                self.outputs[input_path]['port'].append(port_name)
    
    def addDataStore(self, node):
        name = self.core.get_attribute(node,'name')
        path = self.core.get_path(node)
        if name in self.data_store_names:
            return
        self.data_store_names.append(name)
        self.data_stores[path]= {}
        self.data_stores[path]['name'] = name
        self.data_stores[path]['job_ref'] = []
        self.data_stores[path]['data'] = []
        datasets = self.core.get_member_paths(node,'Data')
        if (datasets and len(datasets)>0):
            self.data_stores[path]['data']=datasets
        data_job_path = self.core.get_pointer_path(node, 'Ref')
        if data_job_path:
            data_job_node = self.core.load_by_path(self.root_node, data_job_path)
            if data_job_node:
                workflowname, job_path = self.get_workflow_job_path(data_job_node)
                self.data_stores[path]['job_ref'] = [workflowname, job_path]

                
    
    def processRepeatCheck(self,wnodes):

        for child in wnodes:
            if self.core.get_meta_type(child) == self.meta_("Repeat_Check"):
                repeat_check_path = self.core.get_path(child)
                #self.checkScript = self.core.get_attribute(child,'terminate')
                #self.updateScript = self.core.get_attribute(child,'update')
                self.checkScript[repeat_check_path] = self.core.get_attribute(child,'script')
                self.maxIterations[repeat_check_path] = self.core.get_attribute(child,'max_iteration')
                self.checkScriptPath[repeat_check_path] = ''
                self.repeat_check_src[repeat_check_path]= ''
                self.restartJobName[repeat_check_path] = ''
                continue

        for child in wnodes:
            if self.core.get_meta_type(child) == self.meta_("WF_Check"):
                repeat_check_src_job_path = self.core.get_pointer_path(child, 'src')
                repeat_check_path = self.core.get_pointer_path(child, 'dst')
                self.repeat_check_src_job_path[repeat_check_src_job_path] = repeat_check_path
                self.repeat_check_src[repeat_check_path]=repeat_check_src_job_path
                srcnode =  self.core.load_by_path(self.root_node, repeat_check_src_job_path)
                srcname = self.core.get_attribute(srcnode, 'name')
                srcname = re.sub(' ','_',srcname)
                self.checkScriptPath[repeat_check_path]= os.path.join(self.script_output_dir, srcname+'_check.py')
                continue
            if self.core.get_meta_type(child) == self.meta_("WF_Repeat"):
                rdst = self.core.get_pointer_path(child, 'dst')
                repeat_check_path = self.core.get_pointer_path(child, 'src')
                restartNode = self.core.load_by_path(self.root_node, rdst)
                self.restartJobName[repeat_check_path] = self.core.get_attribute(restartNode, 'name')
                continue
        


    def get_input_job(self,src_node):
        if self.core.get_meta_type(src_node) == self.meta_("DataStore"):
            return src_node
        src_parent = self.core.get_parent(src_node)
        meta_type= self.core.get_meta_type(src_parent)
        if  (not (meta_type == self.meta_("Loop"))):
            return src_parent
        
        connsdst = self.core.get_collection_paths(src_node,'dst')
        for c in connsdst:
            conn = self.core.load_by_path(self.root_node, c)
            npath = self.core.get_pointer_path(conn,'src')
            node = self.core.load_by_path(self.root_node,npath)
            if self.core.get_meta_type(node) == self.meta_("DataStore"):
                return node
            nodeparent = self.core.get_parent(node)
            meta_type= self.core.get_meta_type(nodeparent)
            if  (not (meta_type == self.meta_("Loop"))):
                logger.info("$$$$$$ got nodeparent for source")
                return nodeparent
            else:
                return self.get_input_job(node)
        return ''
            

    def processJobDependency(self, child):
        dst_path = self.core.get_pointer_path(child, 'dst')
        src_path = self.core.get_pointer_path(child, 'src')
        src_node = self.core.load_by_path(self.root_node, src_path)
        dst_node = self.core.load_by_path(self.root_node, dst_path)
        src_node_name = self.core.get_attribute(src_node,'name')
        dst_node_name = self.core.get_attribute(dst_node,'name')

        logger.info('process job dependency: {0} -> {1}'.format(src_node_name, dst_node_name))

        if (self.core.get_meta_type(child) == self.meta_("True_Conn")):
            logger.info("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        
        if (self.core.get_meta_type(child) == self.meta_("False_Conn")):
            logger.info("************************************************")
        
        dst_job = ''
        src_job = ''
        port_name = ''
        is_loop_job = False
        branch = ''
        actual_src_job = ''
        actual_dst_job = ''

        if (self.core.get_meta_type(src_node)== self.meta_("Jobs_Output")):
            logger.info("processJobDependency: got job output")
            src_job = self.core.get_parent(src_node)
        
        if (self.core.get_meta_type(src_node)== self.meta_("DataStore")):
            logger.info("processJobDependency: got data source")
            src_job = src_node
        
        if ((self.core.get_meta_type(src_node)== self.meta_("WF_Output")) or (self.core.get_meta_type(src_node)== self.meta_("False")) or (self.core.get_meta_type(src_node)== self.meta_("True"))):
            logger.info("processJobDependency: WF_output, branch")
            src_job = self.core.get_parent(src_node)
            port_name = self.core.get_attribute(src_node,'name')
            actual_src_job = self.get_input_job(src_node)
        
        if (src_node_name=='False'):
            src_job = self.core.get_parent(src_node)
            logger.info('**************************************** in false branch')
            branch = 'FALSE'
        if (src_node_name=='True'):
            src_job = self.core.get_parent(src_node)
            logger.info('********************************* in true branch')
            branch = 'TRUE'

        
        if (self.core.get_meta_type(src_node)== self.meta_("WF_Input")):
            src_job = self.core.get_parent(src_node)
            port_name = self.core.get_attribute(src_node,'name')
            is_loop_job = True
            actual_src_job = self.get_input_job(src_node)

        if ((self.core.get_meta_type(dst_node)== self.meta_("Jobs_Input")) or (self.core.get_meta_type(dst_node)== self.meta_("WF_Input"))):
            self.addInputSrc(dst_node, src_job, port_name, is_loop_job,actual_src_job)

        if (self.core.get_meta_type(dst_node)== self.meta_("WF_Output")):
            self.addOutputSrc(dst_node, src_job, port_name)
        
        
        dst_job = self.core.get_parent(dst_node)
        if (branch):
            dst_job = dst_node
        if (src_job and dst_job):
            self.addJobDependency(src_job, dst_job, is_loop_job,branch,actual_src_job)

    def processStartJobs(self,startnode):
        if (startnode in self.start_nodes):
            return
        self.start_nodes.append(startnode)
        connssrc = self.core.get_collection_paths(startnode,'src')
        for c in connssrc:
            conn = self.core.load_by_path(self.root_node, c)
            npath = self.core.get_pointer_path(conn,'dst')
            node = self.core.load_by_path(self.root_node,npath)
            nodename = self.core.get_attribute(node,'name')
            if (npath not in self.workflow_start_jobs):
                self.workflow_start_jobs.append(npath)

        
    
    def processWorkflow(self):
        print('1')
        wnodes = self.core.load_sub_tree(self.active_node)
        print('2a')
        for child in wnodes:
            meta_type= self.core.get_meta_type(child)
            if (meta_type == self.meta_("DataStore")):
                self.addDataStore(child)

        print('2b')
        for child in wnodes:
            meta_type= self.core.get_meta_type(child)
            if (meta_type == self.meta_("DataStore")):
                continue
            if ((meta_type == self.meta_("WorkflowJob")) or (meta_type == self.meta_("Loop")) or (meta_type == self.meta_("Transform")) or (meta_type == self.meta_("Branch"))):
                isloop = self.addJob(child)
                self.addActivities(child)
                self.addInputs(child)
                self.addInits(child)
                self.addProperties(child)
                self.addOutputs(child)
        print('3')
        for child in wnodes:
            if ((self.core.get_meta_type(child) == self.meta_("WFExecutionSeq")) or  (self.core.get_meta_type(child) == self.meta_("True_Conn")) or  (self.core.get_meta_type(child) == self.meta_("False_Conn"))):
                self.processJobDependency(child)
            if (self.core.get_meta_type(child) == self.meta_("WF_Start")):
                parent = self.core.get_parent(child)
                if (self.core.get_meta_type(parent) == self.meta_("Workflow")):
                    self.processStartJobs(child)

                
        
        print('4')
        for child in wnodes:
            if self.core.get_meta_type(child) != self.meta_("WorkflowJob") and self.core.get_meta_type(child) != self.meta_("Loop") and self.core.get_meta_type(child) != self.meta_("Transform") and self.core.get_meta_type(child) != self.meta_("Branch"):
                continue
            self.processInputs(child)
            self.processInits(child)
        print('5')
        for child in wnodes:
            if self.core.get_meta_type(child) != self.meta_("Status"):
                continue
            self.status_node = child
        
        print('6')
        self.orderJobs()
        #self.processRepeatCheck(wnodes)
        print('7')
        self.createWorkflowJson()
        print('8')
        
        self.dumpWorkflowJson()
        print('9')
        
    def dumpWorkflowJson(self):
        if (not os.path.exists(self.workflowjson['buildRoot'])):
            os.makedirs(self.workflowjson['buildRoot'])
        
        if (not os.path.exists(self.output_dir)):
            os.makedirs(self.output_dir)
        
        self.createFolder()
        self.workflowjson['working_dir']=self.workflow_ouput_dir

        if self.prototypes:
            self.process_prototypes()

        outputname = os.path.join(self.output_dir,self.workflowname+'.json')
        logger.info(' outputname "{0}" '.format(outputname))
        
        logger.info(' json "{0}" '.format(self.workflowjson))

        with open(outputname, 'w') as outfile:
            json.dump(self.workflowjson, outfile, indent=4, sort_keys=True)
        logger.info('done outputname "{0}" '.format(outputname))

        outputname = os.path.join(self.workflow_ouput_dir,self.workflowname+'.json')
        logger.info(' outputname "{0}" '.format(outputname))
        
        with open(outputname, 'w') as outfile:
            json.dump(self.workflowjson, outfile, indent=4, sort_keys=True)
        logger.info('done outputname "{0}" '.format(outputname))



        
        
        #if (not self.repeat_check_path):
        #    return
        keys = self.checkScript.keys()
        if (len(keys)==0):
            return

        #scriptfoldername = os.path.join(self.output_dir,self.workflowname+'_script')
        #if (not os.path.exists(scriptfoldername)):
        #    os.makedirs(scriptfoldername)
        
        
        #scripttowrite = checkscript
        #scripttowrite = scripttowrite.replace('<<check_script>>', self.checkScript)
        #scripttowrite = scripttowrite.replace('<<update_script>>', self.updateScript)
        #scripttowrite = scripttowrite.replace('<<max_iterations>>', self.maxIterations)
        
        for k in keys:
            scripttowrite = self.checkScript[k]
            checkscriptname = self.checkScriptPath[k]
            f = open (checkscriptname, 'w')
            f.write(scripttowrite)
            f.close()
    

    def process_prototypes(self):
        current_path = str(Path(__file__).absolute().parent.parent.parent)
        print (current_path)
        launch_activity_path=os.path.join(current_path,'LaunchActivity')
        sys.path.append(launch_activity_path)
        prototype_dir = os.path.join(self.workflow_ouput_dir, 'Prototype')
        if not os.path.isdir(prototype_dir):
            os.makedirs(prototype_dir)
        import LaunchActivity
        config = {}
        config["gen_folder"] = prototype_dir
        config["deploy_job"] = False
        for p in self.prototypes.keys():
            active_node_path = self.prototypes[p]
            l = LaunchActivity.LaunchActivity(self.webgme, self.commit_hash, self.branch_name, active_node_path, None, self.namespace,config)
            print ('came here')
            l.main()
            print ('generated')


        
        

            

    def main(self):
        try:
            core = self.core
            root_node = self.root_node
            active_node = self.active_node
            self.config = self.get_current_config()
            logger.info(' self.config "{0}" '.format(str(self.config)))
            print('a')
            if self.core.get_meta_type(active_node) != self.meta_("Workflow"):
                logger.error("Node needs to be a Workflow node")
                raise ValueError("Node needs to be a Workflow node")

            print('b')
            root_children = self.core.load_children(self.root_node)
            for c in root_children:
                if self.core.get_meta_type(c) != self.meta_("ALC"):
                    continue
                self.alcpath = self.core.get_path(c)
                break
            print('c')
            self.workflowname = core.get_attribute(active_node, 'name')
            print('d')
            self.processWorkflow()
            print('e')
            
            #core.set_attribute(active_node, 'name', 'newName')

            commit_info = self.util.save(root_node, self.commit_hash, 'master', 'WorkExecution - '+self.exec_name)
            self.result_set_success(True)
            #logger.info('committed :{0}'.format(commit_info))
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
            self.result_set_error('Workflow Plugin: Error encountered.  Check result details.')
            self.result_set_success(False)
            exit()
