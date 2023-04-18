/*globals define*/
/*jshint node:true, browser:true*/

/**
 * Generated by PluginGenerator 1.7.0 from webgme on Fri Apr 29 2016 10:56:25 GMT-0500 (Central Daylight Time).
 */

define([
    'plugin/PluginConfig',
    'text!./metadata.json',
    'plugin/PluginBase',
    './meta',
	'q',
	'./GSNIDGENUserChoiceDialog',
	'text!./GSNIDGENUserChoiceDialog.html'
], function (
    PluginConfig,
    pluginMetadata,
    PluginBase,
    MetaTypes,
	Q,
	UserChoiceDialog,
	UserChoiceDialogTemplate) {
    'use strict';

    pluginMetadata = JSON.parse(pluginMetadata);

    /**
     * Initializes a new instance of GSNIDGen.
     * @class
     * @augments {PluginBase}
     * @classdesc This class represents the plugin GSNIDGen.
     * @constructor
     */
    var GSNIDGen = function () {
        // Call base class' constructor.
        PluginBase.call(this);
        this.pluginMetadata = pluginMetadata;
		this.metaTypes = MetaTypes;
		this.nodes={};
		this.goalID=0;
		this.strategyID=0;
		this.solutionID=0;
		this.contextID=0;
		this.assumptionID=0;
		this.justificationID=0;
		this.nodesAssigned=[];
		this.gsnModelsToVisit=[];
		this.gsnModelsVisited=[];
    };

    /**
     * Metadata associated with the plugin. Contains id, name, version, description, icon, configStructue etc.
     * This is also available at the instance at this.pluginMetadata.
     * @type {object}
     */
    GSNIDGen.metadata = pluginMetadata;

    // Prototypical inheritance from PluginBase.
    GSNIDGen.prototype = Object.create(PluginBase.prototype);
    GSNIDGen.prototype.constructor = GSNIDGen;

	GSNIDGen.prototype.compareObjs = function (a, b) {
		var self = this;
		if (a.pid != b.pid){
			return a.pid - b.pid;
		}
		var aposition= self.core.getRegistry(a.node, 'position');
		var bposition= self.core.getRegistry(b.node, 'position');
		var diffy=aposition.y - bposition.y;
		var diffx=aposition.x - bposition.x;
		if (diffy==0)
			return diffx;
		return diffy;
	};
	
	GSNIDGen.prototype.getDestinationNodes = function (connNodes) {
	    
	    var self = this, i,j;
		var dstNodes=[];
		self.logger.debug(' in get destination nodes');
		if (!self.core)
		{
			self.logger.debug('nul core');
			return dstNodes;
		}
		for (i = 0; i < connNodes.length; i += 1) {
			var connnode= self.nodes[connNodes[i]];
			if (!connnode)
				continue;
			var dstPath = self.core.getPointerPath(connnode,'dst');
			self.logger.debug(' dst path '+dstPath);
			if (!dstPath)
				continue;
			var dstNode=self.nodes[dstPath];
			if (!dstNode)
				continue;
                        self.logger.debug(' got dst node ');				
			if (self.isMetaTypeOf(dstNode,self.META.ChoiceJn))
			{
				
				var connNodesChoiceJn=self.core.getCollectionPaths(dstNode, 'src');
				var dstNodesChoiceJn = self.getDestinationNodes(connNodesChoiceJn);
				for (j=0; j!=dstNodesChoiceJn.length; j+=1)
				{
					dstNodes.push(dstNodesChoiceJn[j]);
				}
				
			}
			else
			{	self.logger.debug(' meta type - not choice ');		
				if (self.isMetaTypeOf(dstNode,self.META.SupportRef))
				{
					self.logger.debug(' meta type - ref ');		
					var dstPathRef = self.core.getPointerPath(dstNode,'Ref');
					if (dstPathRef)
					{
						var dstRef = self.nodes[dstPathRef];
						if (dstRef)
						{
							dstNodes.push(dstRef);
						}
					}
					
				}
				else
				{
					self.logger.debug(' meta type - not ref ');		
					self.logger.debug(' pushing node '+dstPath);
					dstNodes.push(dstNode);
				}
				self.logger.debug(' ?????meta type - not choice ');	
			}
		}
		return dstNodes;
	};
		
	
	GSNIDGen.prototype.traverseAndAssignIDs = function (callback, goalobjs,stratobjs) {
	    var self = this, i,j, parentID;
		
		
		var strategyobjs=[];
		var solutionobjs=[];
		var contextobjs=[];
		var assumptionobjs=[];
		var justificationobjs=[];
		var nextgoalobjs=[];
		
		self.goalID = self.assignID(self.goalID,goalobjs);
		self.strategyID = self.assignID(self.strategyID,stratobjs);
		
		
		var g=[];
		g=g.concat(goalobjs,stratobjs);
		
		
		for (i = 0; i < g.length; i += 1) {
			var nodeObject = g[i].node;
			parentID =  Number(self.core.getAttribute(nodeObject, 'gsnid'));
			self.logger.debug('load connection');
			var connNodes=self.core.getCollectionPaths(nodeObject, 'src');
			self.logger.debug('finished load connection');
			var dstNodes = self.getDestinationNodes(connNodes);
			 
			for (j = 0; j < dstNodes.length; j += 1) {
				
				var dstNode = dstNodes[j];
				if (self.nodesAssigned.indexOf(dstNode)>=0)
					continue;
					
				var metatype = self.getMetaType(dstNode);
				var name=self.core.getAttribute(dstNode,'name');
				self.logger.debug('obj1 ' +name);
				if (self.isMetaTypeOf(dstNode,self.META.Goal))
				{
					nextgoalobjs.push({pid:parentID,node:dstNode});
				}
				if (self.isMetaTypeOf(dstNode,self.META.Strategy))
				{
					strategyobjs.push({pid:parentID,node:dstNode});
				}
				if (self.isMetaTypeOf(dstNode,self.META.Solution))
				{
					solutionobjs.push({pid:parentID,node:dstNode});
				}
				if (self.isMetaTypeOf(dstNode,self.META.Context))
				{
					contextobjs.push({pid:parentID,node:dstNode});
				}
				if (self.isMetaTypeOf(dstNode,self.META.Assumption))
				{
					assumptionobjs.push({pid:parentID,node:dstNode});
				}
				if (self.isMetaTypeOf(dstNode,self.META.Justification))
				{
					justificationobjs.push({pid:parentID,node:dstNode});
				}
				
				if (self.isMetaTypeOf(dstNode,self.META.GSN_Model))
				{
					self.gsnModelsToVisit.push({pid:parentID,node:dstNode});
				}
			}
		}
		
		self.strategyID = self.assignID(self.strategyID,strategyobjs);
		self.solutionID = self.assignID(self.solutionID,solutionobjs);
		self.contextID = self.assignID(self.contextID,contextobjs);
		self.assumptionID = self.assignID(self.assumptionID,assumptionobjs);
		self.justificationID = self.assignID(self.justificationID,justificationobjs);
		
		solutionobjs=[];
		contextobjs=[];
		assumptionobjs=[];
		justificationobjs=[];
		
		
		for (i = 0; i < strategyobjs.length; i += 1) {
			var nodeObject = strategyobjs[i].node;
			var name=self.core.getAttribute(nodeObject,'name');
			self.logger.debug('strategy ' +name);
			parentID =  Number(self.core.getAttribute(nodeObject, 'gsnid'));
			self.logger.debug('parent id '+parentID);
			var connNodes=self.core.getCollectionPaths(nodeObject, 'src');
			var dstNodes = self.getDestinationNodes(connNodes);
        	for (j = 0; j < dstNodes.length; j += 1) {
				
				var dstNode = dstNodes[j];
				if (self.nodesAssigned.indexOf(dstNode)>=0)
					continue;
				var metatype = self.getMetaType(dstNode);
				var name=self.core.getAttribute(dstNode,'name');
				self.logger.debug('obj2 ' +name);
				if (self.isMetaTypeOf(dstNode,self.META.Goal))
				{
					nextgoalobjs.push({pid:parentID,node:dstNode});
				}
				if (self.isMetaTypeOf(dstNode,self.META.Solution))
				{
					solutionobjs.push({pid:parentID,node:dstNode});
				}
				if (self.isMetaTypeOf(dstNode,self.META.Context))
				{
					contextobjs.push({pid:parentID,node:dstNode});
				}
				if (self.isMetaTypeOf(dstNode,self.META.Assumption))
				{
					assumptionobjs.push({pid:parentID,node:dstNode});
				}
				if (self.isMetaTypeOf(dstNode,self.META.Justification))
				{
					justificationobjs.push({pid:parentID,node:dstNode});
				}
				if (self.isMetaTypeOf(dstNode,self.META.GSN_Model))
				{
					self.gsnModelsToVisit.push({pid:parentID,node:dstNode});
				}
			}
		}
		self.solutionID = self.assignID(self.solutionID,solutionobjs);
		self.contextID = self.assignID(self.contextID,contextobjs);
		self.assumptionID = self.assignID(self.assumptionID,assumptionobjs);
		self.justificationID = self.assignID(self.justificationID,justificationobjs);
		
		if (nextgoalobjs.length!=0)
			self.traverseAndAssignIDs(callback,nextgoalobjs,[]);
    };
	
	GSNIDGen.prototype.assignID = function (idcount, objs) {
		var self = this,i;
		objs.sort(function(a, b) {
		
		if (a.pid != b.pid){
			return a.pid - b.pid;
		}
		var aposition= self.core.getRegistry(a.node, 'position');
		var bposition= self.core.getRegistry(b.node, 'position');
		var diffy=aposition.y - bposition.y;
		var diffx=aposition.x - bposition.x;
		if (diffy==0)
			return diffx;
		return diffy;
	});
		for (i = 0; i < objs.length; i += 1) {
			
			var nodeObject = objs[i].node;
			if (self.nodesAssigned.indexOf(nodeObject)>=0)
				continue;
			idcount +=1;
			var gsnid= idcount.toString();
			var name= self.core.getAttribute(nodeObject,'name');
			self.logger.debug('assign id for'+name + ' ' + gsnid);
			self.core.setAttribute(nodeObject, 'gsnid', gsnid);
			self.nodesAssigned.push(nodeObject);
        }
		return idcount;
    };
	
	GSNIDGen.prototype.traverseGSNModel = function(callback,nodeObject){
		var self = this;
		var i,j, k, metatype,connectionPaths;
		
		self.gsnModelsVisited.push(nodeObject);
		
		
		
		return self.core.loadChildren(nodeObject)
		.then(function(childList)
		{
		
			var gobjs=[];
			var sobjs=[];
			for (j = 0; j < childList.length; j += 1) {
				if (self.core.isConnection(childList[j]))
				{
					continue;
				}
				var name = self.core.getAttribute(childList[j], 'name');
				self.logger.debug(' node ' + name);
				if (self.isMetaTypeOf(childList[j],self.META.Goal))
				{
				
					connectionPaths = self.core.getCollectionPaths(childList[j], 'dst');
					if (connectionPaths.length==0)
					{
						gobjs.push({pid:-1,node:childList[j]});
					}
				}
				
				if (self.isMetaTypeOf(childList[j],self.META.Strategy))
				{
				
					connectionPaths = self.core.getCollectionPaths(childList[j], 'dst');
					if (connectionPaths.length==0)
					{
						
						sobjs.push({pid:-1,node:childList[j]});
					}
				}
			
			}
			self.traverseAndAssignIDs(callback, gobjs,sobjs);
			var gsnmodels=[];
			for (i = 0; i < childList.length; i += 1) {
				if (self.core.isConnection(childList[i]))
				{
					continue;
				}
				var name = self.core.getAttribute(childList[i], 'name');
				metatype = self.getMetaType(childList[i]);
				if (self.isMetaTypeOf(childList[i],self.META.GSN_Model))
				{
					self.logger.debug('adding gsn model' +name);
					gsnmodels.push(childList[i]);
					continue;
				}
				
				var gsnid = self.core.getAttribute(childList[i], 'gsnid');
				self.logger.debug(name + ' '+gsnid);
			}			
			self.logger.debug('gsn models visited length'+ self.gsnModelsVisited.length);
			
			gsnmodels.sort(function(a, b) {
				var aposition= self.core.getRegistry(a, 'position');
				var bposition= self.core.getRegistry(b, 'position');
				var diffy=aposition.y - bposition.y;
				var diffx=aposition.x - bposition.x;
				if (diffy==0)
					return diffx;
				return diffy;
			});
		
			
			var p=[];
			for (i = 0; i < gsnmodels.length; i += 1) {
				var name = self.core.getAttribute(gsnmodels[i], 'name');
				var idx= self.gsnModelsVisited.indexOf(gsnmodels[i]);
				self.logger.debug('Trying to visit gsn model-0 ' +name + ' idx '+ idx);
				if (self.gsnModelsVisited.indexOf(gsnmodels[i]) >=0)
				{
					continue;
				}
				
				self.logger.debug('Trying to visit gsn model' +name);
				p.push(self.traverseGSNModel(callback,gsnmodels[i]));
			}
			return Q.all(p);
				
		});
			
   };
   
   GSNIDGen.prototype.clearIDs = function (nodeObject,userinput) {
		var self = this;
		if (self.isMetaTypeOf(nodeObject,self.META.Goal) || 
		    self.isMetaTypeOf(nodeObject,self.META.Solution) ||
			self.isMetaTypeOf(nodeObject,self.META.Strategy)||
			self.isMetaTypeOf(nodeObject,self.META.Assumption) || 
			self.isMetaTypeOf(nodeObject,self.META.Justification) || 
			self.isMetaTypeOf(nodeObject,self.META.Context))
		{
			self.core.setAttribute(nodeObject, 'gsnid', '');
		}
		
   };
   
   GSNIDGen.prototype.preserveIDs = function (nodeObject,userinput) {
		var self = this;
		var id;
		var name=self.core.getAttribute(nodeObject,'name');
		self.logger.debug('preserving id for ' + name );
		if (self.isMetaTypeOf(nodeObject,self.META.Goal))
		{
			id =  self.core.getAttribute(nodeObject, 'gsnid');
			if (id=='')
				return;
			
			id =  Number(self.core.getAttribute(nodeObject, 'gsnid'));
			if ((!isNaN(id)) && (id>self.goalID))
				self.goalID=id;
			this.nodesAssigned.push(nodeObject);
			
			return;
			
		}
		if (self.isMetaTypeOf(nodeObject,self.META.Strategy))
		{
			id =  self.core.getAttribute(nodeObject, 'gsnid');
			if (id=='')
				return;
			id =  Number(self.core.getAttribute(nodeObject, 'gsnid'));
			if (!isNaN(id) && (id>self.strategyID))
				self.strategyID=id;
			this.nodesAssigned.push(nodeObject);
			return;
		}
		if (self.isMetaTypeOf(nodeObject,self.META.Solution))
		{
			id =  self.core.getAttribute(nodeObject, 'gsnid');
			if (id=='')
				return;
			id =  Number(self.core.getAttribute(nodeObject, 'gsnid'));
			if (!isNaN(id) && (id>self.solutionID))
				self.solutionID=id;
			this.nodesAssigned.push(nodeObject);
			return;
		}
		if (self.isMetaTypeOf(nodeObject,self.META.Context))
		{
			id =  self.core.getAttribute(nodeObject, 'gsnid');
			if (id=='')
				return;
			id =  Number(self.core.getAttribute(nodeObject, 'gsnid'));
			if (!isNaN(id) && (id>self.contextID))
				self.contextID=id;
			this.nodesAssigned.push(nodeObject);
			return;
		}
		if (self.isMetaTypeOf(nodeObject,self.META.Assumption))
		{
			id =  self.core.getAttribute(nodeObject, 'gsnid');
			if (id=='')
				return;
			id =  Number(self.core.getAttribute(nodeObject, 'gsnid'));
			if (!isNaN(id) && (id>self.assumptionID))
				self.assumptionID=id;
			this.nodesAssigned.push(nodeObject);
			return;
		}
		if (self.isMetaTypeOf(nodeObject,self.META.Justification))
		{
			id =  self.core.getAttribute(nodeObject, 'gsnid');
			if (id=='')
				return;
			id =  Number(self.core.getAttribute(nodeObject, 'gsnid'));
			if (!isNaN(id) && (id>self.justificationID))
				self.justificationID=id;
			this.nodesAssigned.push(nodeObject);
			return;
		}
   };
	
	GSNIDGen.prototype.process = function (callback,nodeObject,userinput) {
        var self = this;
		self.logger.debug('4')
		self.core.loadSubTree(nodeObject)
		.then(function(nodeList) {
			var i,j,nodePath;
			self.logger.debug('5');
			self.logger.debug(' nodelist length ' +  nodeList.length);
			
			for (i = 0; i < nodeList.length; i += 1) {
				 nodePath = self.core.getPath(nodeList[i]);
				 self.logger.debug('added nodepath ' + nodePath);
				 self.nodes[nodePath] = nodeList[i];
				if (userinput!=3)
					self.clearIDs(nodeList[i], userinput);
				else
					self.preserveIDs(nodeList[i], userinput);
			}
			self.logger.debug('executing options');
			if (userinput !=1)
			{	
				self.logger.debug('executing options '+userinput);
				self.traverseGSNModel(callback,nodeObject)
				.then(function(){
					self.save('ID Generator updated model.')
					.then(function(){
						self.result.setSuccess(true);
						callback(null, self.result);
					});
				});
			}
			else
			{
				self.save('ID Generator updated model.')
					.then(function(){
						self.result.setSuccess(true);
						callback(null, self.result);
					});
			}
		});
	};
	
    /**
     * Main function for the plugin to execute. This will perform the execution.
     * Notes:
     * - Always log with the provided logger.[error,warning,info,debug].
     * - Do NOT put any user interaction logic UI, etc. inside this method.
     * - callback always has to be called even if error happened.
     *
     * @param {function(string, plugin.PluginResult)} callback - the result callback
     */
    GSNIDGen.prototype.main = function (callback) {
        // Use self to access core, project, result, logger etc from PluginBase.
        // These are all instantiated at this point.
        var self = this,
            nodeObject;
			
		nodeObject = self.activeNode;

	self.logger.debug('1')
        if (self.core.getPath(self.activeNode) === ' ' || self.isMetaTypeOf(self.activeNode, self.META.GSN_Model) === false)
        {
            callback('ActiveNode is not a GSN_Model', self.result);
            return;
        }
	self.logger.debug('2')


        var dialog = new UserChoiceDialog();
		 dialog.show(function (val){
         try {
		 self.logger.debug('3')
			self.logger.debug(val);
			self.process(callback,nodeObject, val);
				
            } catch (e) {
				self.logger.error('user input');
              }
		});

    };

    return GSNIDGen;
});