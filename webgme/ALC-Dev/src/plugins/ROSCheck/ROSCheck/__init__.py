"""
This is where the implementation of the plugin code goes.
The ROSCheck-class is imported from both run_plugin.py and run_debug.py
"""
import sys
import logging
from webgme_bindings import PluginBase
import re

# Setup a logger
logger = logging.getLogger('ROSCheck')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)  # By default it logs to stderr..
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


class ROSCheck(PluginBase):
    def __init__(self, *args, **kwargs):
        super(ROSCheck, self).__init__(*args, **kwargs)
        self.ros_interface_info = {}
        self.interface_keys= ['Publisher','Subscriber','Service Server','Service Client','Action Server','Action Client']
        self.violations = {}
        self.topic_violations = {}
        self.portsVisited = []
        self.topics_to_ports = {}
        self.topics = {}
        self.signalports = []
        self.port_topics = {}
        self.port_topic_keys = []
        self.port_inferredtopic = {}
        self.valid_roles = ['Node', 'Driver', 'Simulation']
        self.active_list = []
        self.inactive_list = []
    
    def updateName(self, name):
        name = re.sub(' ','_',name)
        return name

    
    def getConnectedPorts(self,port, portsVisited):
        #logger.info('**********************')
        #logger.info('port info : '+str(port))
        #logger.info('**********************')
        connsdst = self.core.get_collection_paths(port,'dst')
        connssrc = self.core.get_collection_paths(port,'src')
        ret = []

        for c in connsdst:
            conn = self.core.load_by_path(self.root_node, c)
            if self.core.get_meta_type(conn) == self.META["ALCMeta.SEAM.SignalFlow"]: 
                npath = self.core.get_pointer_path(conn,'src')
                node = self.core.load_by_path(self.root_node,npath)
                if (node not in portsVisited):
                    if (node not in ret):
                        ret.append(node)
        
        for c in connssrc:
            conn = self.core.load_by_path(self.root_node, c)
            if self.core.get_meta_type(conn) == self.META["ALCMeta.SEAM.SignalFlow"]: 
                npath = self.core.get_pointer_path(conn,'dst')
                node = self.core.load_by_path(self.root_node,npath)
                if (node not in portsVisited):
                    if (node not in ret):
                        ret.append(node)
        
        return ret
                    


    def traversePorts(self, port,  portsVisited):
        if (port in portsVisited):
            return
        portsVisited.append(port)
        ports = self.getConnectedPorts(port, portsVisited)
        for p in ports:
            if p not in portsVisited:
                self.traversePorts(p,portsVisited)
    
    def getTopicsTypes(self,ports):
        topics = []
        for p in ports:
            ppath = self.core.get_path(p)
            if (ppath not in self.port_topic_keys):
                continue
            t = self.port_topics[ppath]
            if (t and t not in topics):
                topics.append(t)
        return topics

    def assignInferredTopic(self,ports,topics):
        
        for p in ports:
            ppath = self.core.get_path(p)
            if (ppath not in self.port_topic_keys):
                continue
            t = self.port_topics[ppath]

            inferred_topics = []
            for i in topics:
                if (i == ''):
                    continue
                if (i == t):
                    continue
                if (i not in inferred_topics):
                    inferred_topics.append(i)
            self.port_inferredtopic[ppath]= inferred_topics
            if (len(inferred_topics)):
                istr = '\n'.join(inferred_topics)
                self.core.set_attribute(p,'inferredTopic',istr)
            else:
                self.core.set_attribute(p,'inferredTopic','')

    def addPortTopic(self, t,p):
        ppath = self.core.get_path(p)
        if (p not in self.signalports):
            #logger.info('added to signal ports '+str(p))
            self.signalports.append(p)
            self.port_topics[ppath]=t
            self.port_inferredtopic[ppath]=[]
        if (t == ''):
            return
        keys = self.topics_to_ports.keys()
        if (t not in keys):
            self.topics_to_ports[t] = []
        self.topics_to_ports[t].append(p)
    
    def checkAllTopicPorts(self):
        keys = self.topics_to_ports.keys()
        for t in keys:
            self.checkTopicPorts(t)
    
    
    def checkTopicPorts(self, t):
        ports = self.topics_to_ports[t]
        if (not ports):
            return
        src = []
        dst = []
        srctype = []
        dsttype = []
        messagetype = []
        for p in ports:
            ptype = self.core.get_attribute(p,'PortType')
            if (ptype and ptype in ['Publisher', 'Action Server', 'Service Server']):
                src.append(p)
                if (ptype not in srctype):
                    srctype.append(ptype)
                
                msg_obj = self.core.load_pointer(p,'messagetype')
                if (msg_obj):
                    path = self.core.get_path(msg_obj)
                    if (path not in messagetype):
                        messagetype.append(path)
        
        if (len(srctype)>1):
            self.addTopicViolation(t,'Multiple source types with same topic')
            return
        else:
            if (srctype in ['Action Server', 'Service Server']):
                if (len(src)>1):
                    self.addTopicViolation(t,'Multiple servers for the same topic')
                    return
            
            if (len(messagetype)>1):
                self.addTopicViolation(t,'Multiple message types for the same topic')
                return
        
        sptype = ''
        if (len(srctype)):
            sptype = srctype[0]
        
        for p in ports:
            ptype = self.core.get_attribute(p,'PortType')
            if (not ptype):
                continue
            if (ptype in ['Publisher', 'Action Server', 'Service Server']):
                continue
            if (not sptype):
                self.addViolation(p, 'No source found for topic '+t)
                continue

            if (sptype == 'Publisher' and ptype=='Subscriber'):
                continue
            if (sptype == 'Action Server' and ptype=='Action Client'):
                continue
            if (sptype == 'Service Server' and ptype=='Service Client'):
                continue
            self.addViolation(p,'Interface type mismatch Source : '+sptype)
        
        mtype = ''
        if (len(messagetype)):
            mtype = messagetype[0]
        mnode = self.core.load_by_path(self.root_node,mtype)
        mname = self.core.get_attribute(mnode, 'name')
        for p in ports:
            ptype = self.core.get_attribute(p,'PortType')
            if (ptype in ['Publisher', 'Action Server', 'Service Server']):
                continue
            
            if (not mtype):
                self.addViolation(p,'Source Message type not found for topic: '+t )
                
            if (len(srctype)==0):
                continue

            msg_obj = self.core.load_pointer(p,'messagetype')
            if (msg_obj):
                path = self.core.get_path(msg_obj)
                if (path != mtype):
                    self.addViolation(p,'Message type mismatch. Source for topic '+t + ' set to message type :'+mname)

            
    def addTopicViolation(self, t, issuestr):
        keys = self.topic_violations.keys()
        if (t not in keys):
            self.topic_violations[t]=[]
        self.topic_violations[t].append(issuestr)    

    
    def addViolation(self, p, issuestr):
        ppath = self.core.get_path(p)
        if (ppath not in self.violations):
            self.violations[ppath] = []
        self.violations[ppath].append(issuestr)

    

    # check node names are unique
    # check port type, message type and topic are set
    # check port type and message type match
    # trace through the links and check if the topics are consistent, port types are consistent
    def runNodeCheck(self, node):
        cnodes  = self.core.load_children(node)
        portnames = []
        for c in cnodes:
            if (self.core.get_meta_type(c) == self.META["ALCMeta.SignalPort"]):
                ptype = self.core.get_attribute(c,'PortType')
                if (ptype != 'Other' and ptype != 'Signal'):
                    name  = self.core.get_attribute(c,'name')
                    portnames.append(name)
                else:
                    self.core.set_attribute(c,'issues','')

        for c in cnodes:
            if self.core.get_meta_type(c) == self.META["ALCMeta.SignalPort"]:
                ptype = self.core.get_attribute(c,'PortType')
                if (ptype != 'Other' and ptype != 'Signal'):
                    self.checkPort(c, portnames)

    
    def checkPort(self,p,portnames):
        issues = []
        name  = self.core.get_attribute(p,'name')
        
        ptype = self.core.get_attribute(p,'PortType')
        msg_obj = self.core.load_pointer(p,'messagetype')
        topic = self.core.get_attribute(p,'Topic').strip()
        queue_size = self.core.get_attribute(p,'queueSize')
        
        if (name  == ''):
            issues.append(' Port name is empty')
        else:
            count = portnames.count(name)
            if (count > 1):
                issues.append(' Port name is not unique in the block')

        if (topic == ''):
            issues.append('Topic is not set')
            #self.topic_not_set.append(p)
        
        if ((queue_size < 0) and ((ptype == 'Publisher') or (ptype == 'Subscriber'))):
            issues.append('Queue size should be a positive integer')
        
        if (not msg_obj):
            issues.append('Message type is not set')
        else:
            if ((ptype == 'Publisher') or (ptype == 'Subscriber')):
                if (self.core.get_meta_type(msg_obj) != self.META["ALCMeta.MessageType"]):
                    issues.append('For a publisher/ subcriber, the message type should be a MessageType object')
        
            if ((ptype == 'Action Client') or (ptype == 'Action Server')):
                if (self.core.get_meta_type(msg_obj) != self.META["ALCMeta.ActionType"]):
                    issues.append('For a Action Server/ Client, the message type should be a ActionType object')

            if ((ptype == 'Service Client') or (ptype == 'Service Server')):
                if self.core.get_meta_type(msg_obj) != self.META["ALCMeta.ServiceType"]:
                    issues.append('For a Service Server/ Client, the message type should be a ServiceType object')
        if (not ptype):
            issues.append('Port type is not set')
        else:
            if (ptype in ['Publisher', 'Action Server', 'Service Server']):
                if ((topic) and (topic in self.topic_violations.keys()) and (len(self.topic_violations[topic])>0)):
                    issues.extend(self.topic_violations[topic])

        

        #self.violations[port_path] = issues
        issue_str = '\n'.join(issues)
        self.addViolation(p,issue_str)
        


    def addNodePorts(self, node, type='Node'):
        cnodes  = self.core.load_children(node)
        portnames = []
        for c in cnodes:
            if self.core.get_meta_type(c) == self.META["ALCMeta.SignalPort"]:
                port_direction = self.core.get_attribute(c,'Direction')
                if (type != 'Node' and port_direction == 'Input'):
                    continue
                topic = self.core.get_attribute(c,'Topic')
                self.addPortTopic(topic, c)

    def checkIfNodeOrParentIsActive(self, node):
        if (node in self.active_list):
            return True
        if (node in self.inactive_list):
            return False
        is_active =self.core.get_attribute(node,'IsActive')
        is_impl = self.core.get_attribute(node,'IsImplementation')
        if (is_impl and not is_active):
            self.inactive_list.append(node)
            return False
        parent = self.core.get_parent(node)
        if (self.core.get_meta_type(parent) != self.META["ALCMeta.Block"]):
            self.active_list.append(node)
            return True
        ret = self.checkIfNodeOrParentIsActive(parent)
        if ret: 
            self.active_list.append(node) 
            return True

        self.inactive_list.append(node)
        return False
        
        

    def runSystemCheck(self):
        nodes = self.core.load_sub_tree(self.active_node)
        ros_nodes = []
        
        for n in nodes:
            if (self.core.get_meta_type(n) == self.META["ALCMeta.Block"]):
                if (not self.checkIfNodeOrParentIsActive(n)):
                    continue
                brole = self.core.get_attribute(n,'Role')
                role = self.core.get_attribute(n,'Other_Role')
                if (role in self.valid_roles or brole in self.valid_roles):
                    ros_nodes.append(n)
                    self.addNodePorts(n,brole)
        
        self.port_topic_keys = self.port_topics.keys()
        

        allVisitedPorts = []
        for s in self.signalports:
            if (s in allVisitedPorts):
                continue
            portsvisited = []
            #logger.info('------------------------')
            #logger.info('signal port '+str(s))
            #logger.info('------------------------')
            self.traversePorts(s,portsvisited)
            topics = self.getTopicsTypes(portsvisited)
            self.assignInferredTopic(portsvisited,topics)
            for p in portsvisited:
                if p not in allVisitedPorts:
                    allVisitedPorts.append(p)
        
        self.checkAllTopicPorts()

        for n in ros_nodes:
            self.runNodeCheck(n)
        
    def updatePortIssues(self):
        keys = self.violations.keys()
        for k in keys:
            n = self.core.load_by_path(self.root_node,k)
            self.core.set_attribute(n,'issues','\n '.join(self.violations[k]))
            if (self.violations[k]):
                name = self.core.get_attribute(n,'name')
                #logger.info(' '+name+' : '+'\n '.join(self.violations[k]))
    

    def main(self):
        try:
            core = self.core
            root_node = self.root_node
            active_node = self.active_node
            if not ((self.core.get_meta_type(active_node) == self.META["ALCMeta.Block"]) or (self.core.get_meta_type(active_node) == self.META["ALCMeta.SystemModel"])):
                self.result_set_success(False)
                self.result_set_error('ROSCheck can be run on Block or System Model')
                exit()
                #raise RuntimeError("ROSCheck can be run on Block or System Model")

            if (self.core.get_meta_type(active_node) == self.META["ALCMeta.Block"]):
                role = core.get_attribute(active_node,'Role')
                if (role not in self.valid_roles):
                    self.result_set_success(False)
                    self.result_set_error("Selected Block's role attribute needs to be Node, Driver or Simulation")
                    exit()
                    #raise RuntimeError("Selected Block's (%s) role attribute needs to be Node" % self.active_node)
                self.runNodeCheck(active_node)
            else:
                logger.info('0*************************************')       
                self.runSystemCheck()
                logger.info('0*************************************')
            
            logger.info('1*************************************')
            self.updatePortIssues()
            logger.info('1*************************************')

            self.result_set_success(True)
            commit_info = self.util.save(root_node, self.commit_hash, 'master', 'ROSCheck results saved')
            logger.info('committed :{0}'.format(commit_info))
        except Exception as err:
            self.send_notification(str(err))
            raise err
            # msg = str(err)
            # self.create_message(self.active_node, msg, 'error')
            # self.result_set_error('ROSCheck Plugin: Error encoutered. Check result details.')
            # self.result_set_success(False)
            # exit()
