/*globals define, WebGMEGlobal*/
/*jshint browser: true*/
/**
 * Generated by VisualizerGenerator 1.7.0 from webgme on Thu Jul 07 2016 11:24:16 GMT-0500 (Central Daylight Time).
 */

define(['js/Constants',
    'js/Utils/GMEConcepts',
    'js/NodePropertyNames'
], function (CONSTANTS,
             GMEConcepts,
             nodePropertyNames) {

    'use strict';

    var ParamMapControl;

    ParamMapControl = function (options) {

        this._logger = options.logger.fork('Control');

        this._client = options.client;

        // Initialize core collections and variables
        this._widget = options.widget;
        this._currentNodeId = null;
        this._currentNodeParentId = undefined;
        this.alcid = '';
        this.addCount = 0;
        this._initWidgetEventHandlers();
        this._logger.debug('ctor finished');
    };

    ParamMapControl.prototype._initWidgetEventHandlers = function () {
        var self = this;

        this._widget.onNodeClick = function (id) {
            var targetNodeObj = self._client.getNode(id);
            if (targetNodeObj) {
                var address = window.location.origin + WebGMEGlobal.gmeConfig.client.mountedPath + '/?project=' + encodeURIComponent(self._client.getActiveProjectId());
                address += '&branch=' + encodeURIComponent(self._client.getActiveBranchName());
                address += '&node='+ encodeURIComponent(id);
                address += '&visualizer=Designer';
                window.open(address, '_blank');
                window.focus();
            }

        };

       	
        	
        
        this._widget.onEditParameterInfo = function (id, oldvalue, newvalue) {
            self._logger.debug(' on edit onEditParameterInfo');
            
            if (oldvalue == newvalue)
				return;
            
            self._client.startTransaction('updating parameter table');
            if (oldvalue)
            {
                self._client.removeMember(oldvalue, id, 'linked_objects');

            }
            if (newvalue)
            {
                self._client.addMember(newvalue, id, 'linked_objects');

            }
            
            
            self._client.completeTransaction('updated parameter table', function (err, result) {

                //self._logger.debug(result.hash);
                //self._logger.debug(result.status);
            });
        };

        
        

    };

    /* * * * * * * * Visualizer content update callbacks * * * * * * * */
    // One major concept here is with managing the territory. The territory
    // defines the parts of the project that the visualizer is interested in
    // (this allows the browser to then only load those relevant parts).
    ParamMapControl.prototype.selectedObjectChanged = function (nodeId) {
    
        var desc =  undefined,
                self = this;
        self._getObjectDescriptor(nodeId)
            .then(function(desc) {

                self._logger.debug('activeObject nodeId \'' + nodeId + '\'');
                self._logger.debug('metaname of activeobject= '+ desc.metaName);
        
                // Remove current territory patterns
                if (self._currentNodeId) {
                    self._client.removeUI(self._territoryId);
                }

                self._currentNodeId = nodeId;
                self._currentNodeParentId = undefined;

                if (typeof self._currentNodeId === 'string') {
                    // Put new node's info into territory rules
                    self._selfPatterns = {};
                    self._selfPatterns[nodeId] = {children: 0};  // Territory "rule"

                    self._widget.setTitle("ParamMap");

                    if (typeof desc.parentId === 'string') {
                        self.$btnModelHierarchyUp.show();
                    } else {
                        self.$btnModelHierarchyUp.hide();
                    }

                    self._currentNodeParentId = desc.parentId;

                    self._territoryId = self._client.addUI(self, function (events) {
                        self._eventCallback(events);
                    });

                    // Update the territory
                    self._client.updateTerritory(self._territoryId, self._selfPatterns);

                    self._selfPatterns[nodeId] = {children: 100};
                    /*if (self.alcid)
                    {
                        self._selfPatterns[self.alcid] = { children: 5 };
                    }*/
                    
                    self._client.updateTerritory(self._territoryId, self._selfPatterns);
                }
            });
    };

    // This next function retrieves the relevant node information for the widget
    ParamMapControl.prototype._getObjectDescriptor = function (nodeId) {
        var self=this;
        var nodeObj = self._client.getNode(nodeId);
        var obj, k;
            
	
        return new Promise(function(resolve,reject) {
	
            if (nodeObj) {
                var objDescriptor = {
                    id: undefined,
                    parentId: undefined,
                    metaName: undefined,
                    name: undefined,
					phname: undefined,
                    parameters : {},
                    systemparameters: {},
                    isROSNode : 0,
                    nodetype: '',
					use: 0
                };
                
                var metaObj = self._client.getNode(nodeObj.getMetaTypeId());
				objDescriptor.id = nodeObj.getId();
				objDescriptor.name = nodeObj.getAttribute(nodePropertyNames.Attributes.name);
				objDescriptor.parentId = nodeObj.getParentId();
				
                if (metaObj) {
                    var metaname  = metaObj.getAttribute(nodePropertyNames.Attributes.name);
                    objDescriptor.metaName =  metaname;

                    if (!self._currentNodeId && !(self.alcid))
                    {
                        //self.getParentFolders(nodeObj);
                        resolve(objDescriptor);
                       
                    }
                    else{
                   
                        if ((self._currentNodeId) &&(objDescriptor.metaName == 'ROSInfo'))
                        {
                            
                            var childrenIds = nodeObj.getChildrenIds();
                            var incontext = 0;
                            var phname  = self.getHierName(objDescriptor.id);
                            if (phname == '')
                            {
                                resolve(objDescriptor);
                            }
                            var parentObj = self._client.getNode(objDescriptor.parentId);
                            var noderole = parentObj.getAttribute('Role');
                            var isrosnode = ((noderole=='Node') || (noderole=='Simulation')|| (noderole=='Driver')|| (noderole=='Simulation Component')|| (noderole=='Node Bridge'));
                            if (!isrosnode)
                            {
                                resolve(objDescriptor);
                            }

                            if (isrosnode && phname != '')
                            {
                            
                                objDescriptor.phname = phname;
                                objDescriptor.parameters = {};
                                objDescriptor.isROSNode = 1;
                                objDescriptor.use = 1;
                                //objDescriptor.nodetype = noderole;
                                
                                for (k=0; k<childrenIds.length; k++)
                                {
                                    var cnode = self._client.getNode(childrenIds[k]);
                                    if (!cnode)
                                    {
                                        resolve(objDescriptor);
                                    }
                                    var cmetaObj = self._client.getNode(cnode.getMetaTypeId());
                                    var mname = cmetaObj.getAttribute(nodePropertyNames.Attributes.name);
                                    var cname = cnode.getAttribute(nodePropertyNames.Attributes.name);
                                    var cid  = childrenIds[k];
                                    if (mname == 'ROSArgument')
                                    {
                                        var ptrid = cnode.getPointerId('linked_port');
                                        if (ptrid)
                                        {
                                            continue;
                                        }
                                        var defval =   cnode.getAttribute('default');
                                        
                                        objDescriptor.parameters[cid]={'name':cname,'default':defval}
                                    }
                                        
                                }
                            }
                            
                        }

                        if ((self._currentNodeId) && (objDescriptor.metaName=="Params")) {                        
                            var parentId = objDescriptor.parentId;
                            var parentObj = '';
                            var pMetaName = '';
                            
                            if (parentId) {
                                parentObj = self._client.getNode(parentId);
                                if (parentObj) {

                                    var pmetaObj = self._client.getNode(parentObj.getMetaTypeId());
                                    if (pmetaObj)
                                        pMetaName = pmetaObj.getAttribute(nodePropertyNames.Attributes.name);
                                        if (pMetaName == 'SystemModel')
                                        {
                                            
                                            var childrenIds = nodeObj.getChildrenIds();
                                            var k = 0;
                                            for (k=0; k<childrenIds.length; k++)
                                            {
                                                var cnode = self._client.getNode(childrenIds[k]);
                                                if (!cnode)
                                                {
                                                    resolve(objDescriptor);
                                                }
                                                var cmetaObj = self._client.getNode(cnode.getMetaTypeId());
                                                var mname = cmetaObj.getAttribute(nodePropertyNames.Attributes.name);
                                                var cname = cnode.getAttribute(nodePropertyNames.Attributes.name);
                                                var cid  = childrenIds[k];
                                                if (mname == 'parameter')
                                                {
                                                    var defval =   cnode.getAttribute('value');
                                                    var ids = cnode.getMemberIds('linked_objects') ;
                                                    objDescriptor.systemparameters[cid]={'name':cname,'value':defval, 'rosargs':ids};

                                                }
                                                    
                                            }
                                            
                                            objDescriptor.use = 1;
                                        }
                                }
                            }
                        }
                    }

                    resolve(objDescriptor);
                }
                else {
                    resolve(objDescriptor);
                }
            }
            else {
                resolve(objDescriptor);
            }
        });
    };

    
    
    ParamMapControl.prototype.getHierName = function (nodeid) {
        var self = this;
		var pid = nodeid;
		var nodeObj;
        var ret='';
        if (nodeid == self._currentNodeId && self._currentNodeParentId)
        {
            nodeObj = self._client.getNode(nodeid);
            ret = nodeObj.getAttribute(nodePropertyNames.Attributes.name);
            return ret;
        }
		while (pid != self._currentNodeId)
		{
			nodeObj = self._client.getNode(pid);
			if (nodeObj)
			{	
				if (ret)
				{
				  var ret1 = ret;
				  ret = nodeObj.getAttribute(nodePropertyNames.Attributes.name);
				  ret += '/';
                  ret += ret1; 
					
				}
                else
                {
                    ret = nodeObj.getAttribute(nodePropertyNames.Attributes.name);
                }
				pid = nodeObj.getParentId();
            }
            else
            {
                return '';
            }
        
			
        }
        if (pid == self._currentNodeId)
            return ret;
        
        return '';
        
    };

    ParamMapControl.prototype.getParentFolders = function (nodeObj) {

        var self = this;
        var ret = [];
        if (self.alcid )
            return ret;

        //self._logger.debug('******* in getParentFolders');

        var parentNodeID, parentNode, parentMetaNode, parentMetaName;
        var alcMetaType = 'ALC';
        parentNodeID = nodeObj.getParentId();
        parentNode = self._client.getNode(parentNodeID);

        var metaobj = self._client.getNode(nodeObj.getMetaTypeId());
        var metaName = metaobj.getAttribute(nodePropertyNames.Attributes.name);
        

        while (parentNode) {
            parentMetaNode = self._client.getNode(parentNode.getMetaTypeId());
            if (!parentMetaNode)
                break;

            parentMetaName = parentMetaNode.getAttribute(nodePropertyNames.Attributes.name);

            if (parentMetaName.indexOf(alcMetaType) != -1) {
                self.alcid = parentNodeID;
                break;
            }

            
            parentNodeID = parentNode.getParentId();
            parentNode = self._client.getNode(parentNodeID);

        }

    };



    /* * * * * * * * Node Event Handling * * * * * * * */
    ParamMapControl.prototype._eventCallback = function (events) {
        var i = events ? events.length : 0,
            event;

        //this._logger.debug('_eventCallback \'' + i + '\' items');

        while (i--) {
            event = events[i];
            switch (event.etype) {
                case CONSTANTS.TERRITORY_EVENT_LOAD:
                    this._onLoad(event.eid,i);
                    break;
                case CONSTANTS.TERRITORY_EVENT_UPDATE:
                    this._onUpdate(event.eid);
                    break;
                case CONSTANTS.TERRITORY_EVENT_UNLOAD:
                    this._onUnload(event.eid);
                    break;
                default:
                    break;
            }
        }

        //this._logger.debug('_eventCallback \'' + events.length + '\' items - DONE');
    };

    ParamMapControl.prototype._onLoad = function(gmeId,count=-1){
		var self= this;
		 this._getObjectDescriptor(gmeId)
          .then(function(description) {
              description.last=-1;
              if (count==1){
                  //if (self.addCount !=0)
                  {
                    description.last=1;
                  }

                  
              }
              //self._logger.debug('metaname  _onLoad= '+ description.faultlabel);
              //if ((description.last==1)||((description.use ==1 ) && ((description.metaName == 'Block')||(description.metaName == 'Result')|| (description.metaName == "Params"))))
              if ((description.last==1)|| ((description.use ==1 ) && ((description.metaName == 'ROSInfo')||(description.metaName == 'Params'))))
			  {
                //self._logger.debug('metaname ' + description.metaName + ' hname  \'' + description.phname );
                 self._widget.addNode(description);
                 //self.addCount=1;
			  }
          });
		  
        
    };

        ParamMapControl.prototype._onUpdate = function (gmeId) {
            var self=this;
            self._getObjectDescriptor(gmeId)
            .then(function(description) {
                //self._logger.debug('metaname  _onUpdate= '+ description.metaName);
                self._widget.updateNode(description);
            });
        };

        ParamMapControl.prototype._onUnload = function (gmeId) {
            this._widget.removeNode(gmeId);
        };

        ParamMapControl.prototype._stateActiveObjectChanged = function (model, activeObjectId) {
            if (this._currentNodeId === activeObjectId) {
                // The same node selected as before - do not trigger
            } else {
                this.selectedObjectChanged(activeObjectId);
            }
        };

        /* * * * * * * * Visualizer life cycle callbacks * * * * * * * */
        ParamMapControl.prototype.destroy = function () {
            this._detachClientEventListeners();
            this._removeToolbarItems();
        };

        ParamMapControl.prototype._attachClientEventListeners = function () {
            this._detachClientEventListeners();
            WebGMEGlobal.State.on('change:' + CONSTANTS.STATE_ACTIVE_OBJECT, this._stateActiveObjectChanged, this);
        };

        ParamMapControl.prototype._detachClientEventListeners = function () {
            WebGMEGlobal.State.off('change:' + CONSTANTS.STATE_ACTIVE_OBJECT, this._stateActiveObjectChanged);
        };

        ParamMapControl.prototype.onActivate = function () {
            this._attachClientEventListeners();
            this._displayToolbarItems();

            if (typeof this._currentNodeId === 'string') {
                WebGMEGlobal.State.registerSuppressVisualizerFromNode(true);
                WebGMEGlobal.State.registerActiveObject(this._currentNodeId);
                WebGMEGlobal.State.registerSuppressVisualizerFromNode(false);
            }
        };

        ParamMapControl.prototype.onDeactivate = function () {
            this._detachClientEventListeners();
            this._hideToolbarItems();
        };

        /* * * * * * * * * * Updating the toolbar * * * * * * * * * */
        ParamMapControl.prototype._displayToolbarItems = function () {

            if (this._toolbarInitialized === true) {
                for (var i = this._toolbarItems.length; i--;) {
                    this._toolbarItems[i].show();
                }
            } else {
                this._initializeToolbar();
            }
        };

        ParamMapControl.prototype._hideToolbarItems = function () {

            if (this._toolbarInitialized === true) {
                for (var i = this._toolbarItems.length; i--;) {
                    this._toolbarItems[i].hide();
                }
            }
        };

        ParamMapControl.prototype._removeToolbarItems = function () {

            if (this._toolbarInitialized === true) {
                for (var i = this._toolbarItems.length; i--;) {
                    this._toolbarItems[i].destroy();
                }
            }
        };

        ParamMapControl.prototype._initializeToolbar = function () {
            var self = this,
                toolBar = WebGMEGlobal.Toolbar;

            this._toolbarItems = [];

            this._toolbarItems.push(toolBar.addSeparator());

            /************** Go to hierarchical parent button ****************/
            this.$btnModelHierarchyUp = toolBar.addButton({
                title: 'Go to parent',
                icon: 'glyphicon glyphicon-circle-arrow-up',
                clickFn: function (/*data*/) {
                    WebGMEGlobal.State.registerActiveObject(self._currentNodeParentId);
                }
            });
            this._toolbarItems.push(this.$btnModelHierarchyUp);
            this.$btnModelHierarchyUp.hide();

            /************** Checkbox example *******************/

            this.$cbShowConnection = toolBar.addCheckBox({
                title: 'toggle checkbox',
                icon: 'gme icon-gme_diagonal-arrow',
                checkChangedFn: function (data, checked) {
                    self._logger.debug('Checkbox has been clicked!');
                }
            });
            this._toolbarItems.push(this.$cbShowConnection);

            this._toolbarInitialized = true;
        };

        return ParamMapControl;
    });
