/*globals define, WebGMEGlobal*/
/**
 * Generated by VisualizerGenerator 1.7.0 from webgme on Fri Feb 23 2018 11:40:01 GMT-0600 (Central Standard Time).
 */

define([
    './CONSTANTS'
], function (CONSTANTS) {

    'use strict';

    function ParamsControl(options) {
        this.logger = options.logger.fork('Control');
        const logger = this.logger;
        this._client = options.client;
        const client = this._client;
        

        // Initialize core collections and variables
        this._widget = options.widget;

        this._widget.notifyUser = (msg) => {
            client.notifyUser(msg);
        };

        this._widget.addNewAttribute = (name, description, value) => {
            const attrName = `${CONSTANTS.ATTR_PREFIX}${name}`;

            if (!attrName || attrName.indexOf('.') > 0 || attrName.indexOf('$') > 0) {
                client.notifyUser({severity: 'error', message: 'Invalid attribute name, cannot contain . or $.'});
                return;
            }

            const nodeObj = client.getNode(CONSTANTS.NODE_ID);

            if (nodeObj) {

                client.startTransaction();

                try {
                    if (nodeObj.getValidAttributeNames().indexOf(attrName) > -1) {
                        throw new Error('Asset name is already taken, provide another one.');
                    }

                    client.setAttributeMeta(CONSTANTS.NODE_ID, attrName, {
                        type: 'asset',
                        hidden: true,
                        description: description || '',
                    });

                    if (value) {
                        client.setAttribute(CONSTANTS.NODE_ID, attrName, value);
                    }

                    client.notifyUser({severity: 'success',
                        message: 'Created new asset [' + name + '], edit content and description in table.'});
                } catch (err) {
                    logger.error(err);
                    client.notifyUser({severity: 'error', message: err.message});
                }

                client.completeTransaction();
            }
        };

        this._widget.updateAttributeDescription = (name, newDescription) => {
            try {
                const nodeObj = client.getNode(CONSTANTS.NODE_ID);
                const attrName = `${CONSTANTS.ATTR_PREFIX}${name}`;

                if (nodeObj) {
                    const attrDesc = nodeObj.getAttributeMeta(attrName);
                    attrDesc.description = newDescription;

                    client.setAttributeMeta(CONSTANTS.NODE_ID, `${CONSTANTS.ATTR_PREFIX}${name}`, attrDesc);
                }
            } catch (err) {
                logger.error(err);
                client.notifyUser({severity: 'error', message: err.message});
            }
        };

        this._widget.setAttributeAsset = (name, value) => {
            try {
                client.setAttribute(CONSTANTS.NODE_ID, `${CONSTANTS.ATTR_PREFIX}${name}`, value);
            } catch (err) {
                logger.error(err);
                client.notifyUser({severity: 'error', message: err.message});
            }
        };

        this._widget.renameAttribute = (name, newName) => {
            const attrName = `${CONSTANTS.ATTR_PREFIX}${name}`;
            const newAttrName = `${CONSTANTS.ATTR_PREFIX}${newName}`;
            const nodeObj = client.getNode(CONSTANTS.NODE_ID);

            if (!newName || newName.indexOf('.') > 0 || newName.indexOf('$') > 0) {
                client.notifyUser({severity: 'error', message: 'Invalid attribute name, cannot contain . or $.'});
                return;
            }

            if (nodeObj) {
                const attrDesc = nodeObj.getAttributeMeta(attrName);

                client.startTransaction();
                try {
                    if (nodeObj.getValidAttributeNames().indexOf(newAttrName) > -1) {
                        throw new Error('Asset name is already taken, provide another one.');
                    }
                    client.delAttributeMeta(CONSTANTS.NODE_ID, attrName);
                    client.setAttributeMeta(CONSTANTS.NODE_ID, newAttrName, attrDesc);
                    // TODO: Add this when the client has it on its API
                    // client.renameAttributeMeta(attrName, newAttrName);

                    client.renameAttribute(CONSTANTS.NODE_ID, attrName, newAttrName);
                }
                catch (err) {
                    logger.error(err);
                    client.notifyUser({severity: 'error', message: err.message});
                }

                client.completeTransaction();
            }
        };

        this._widget.deleteAttribute = (name) => {
            const attrName = `${CONSTANTS.ATTR_PREFIX}${name}`;

            client.startTransaction();

            try {
                client.delAttributeMeta(CONSTANTS.NODE_ID, attrName);
                client.delAttribute(CONSTANTS.NODE_ID, attrName);
            } catch (err) {
                logger.error(err);
                client.notifyUser({severity: 'error', message: err.message});
            }

            client.completeTransaction();
        };

        this._attributes = [];
        this._attributesCompare = JSON.stringify(this._attributes);
        this._currentNodeId= null;
        this._currentNodeParentId = null;
        this.parentFoldersVisited = 0;
        this.nodesToVisit = {};

        //this._initWidgetEventHandlers();

/*
        this._uiId = this._client.addUI(null, (events) => {
            const nodeObj = client.getNode(CONSTANTS.NODE_ID);
            let newAttributes = [];

            if (nodeObj) {
                newAttributes = nodeObj.getValidAttributeNames()
                    .filter(attrName => attrName.startsWith(CONSTANTS.ATTR_PREFIX))
                    .sort()
                    .map((attrName) => {
                        return {
                            name: attrName.substring(CONSTANTS.ATTR_PREFIX.length),
                            value: nodeObj.getAttribute(attrName),
                            desc: nodeObj.getAttributeMeta(attrName)
                        }
                    });
            }

            // The widget takes care of only updating what changed..
            const newAttributesStr = JSON.stringify(newAttributes);

            if (this._attributesCompare !== newAttributesStr) {
                this._attributes = newAttributes;
                this._attributesCompare = newAttributesStr;
                this._widget.atNewAttributes(newAttributes);
            }
        });

        const territory = {};
        territory[CONSTANTS.NODE_ID] = {children: 0};

        client.updateTerritory(this._uiId, territory);
        */
    };

/*
    ParamsControl.prototype.onActivate = function () {
        this.logger.debug('ParamsWidget has been activated');
    };

    ParamsControl.prototype.onDeactivate = function () {
        this.logger.debug('ParamsWidget has been deactivated');
    };
*/
    /* * * * * * * * Visualizer life cycle callbacks * * * * * * * */
/*    ParamsControl.prototype.destroy = function () {
        this._client.removeUI(this._uiId);
    };

*/
    /* * * * * * * * Visualizer content update callbacks * * * * * * * */
        // One major concept here is with managing the territory. The territory
        // defines the parts of the project that the visualizer is interested in
        // (this allows the browser to then only load those relevant parts).
        ParamsControl.prototype.selectedObjectChanged = function (nodeId) {

            var desc = undefined,
                self = this;
            self._getObjectDescriptor(nodeId)
                .then(function (desc) {

                    self._logger.debug('activeObject nodeId \'' + nodeId + '\'');
                    self._logger.debug('metaname of activeobject= ' + desc.metaName);

                    // Remove current territory patterns
                    if (self._currentNodeId) {
                        self._client.removeUI(self._territoryId);
                    }

                    self._currentNodeId = nodeId;
                    self._currentNodeParentId = undefined;

                    if (typeof self._currentNodeId === 'string') {
                        // Put new node's info into territory rules
                        self._selfPatterns = {};
                        self._selfPatterns[nodeId] = { children: 2 };  // Territory "rule"

                        self._widget.setTitle("");

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
                        //self._client.updateTerritory(self._territoryId, self._selfPatterns);

                        //self._selfPatterns[nodeId] = {children: 100};
                        //self._client.updateTerritory(self._territoryId, self._selfPatterns);

                        self._selfPatterns[nodeId] = { children: 2 };

                        if (self.nodesToVisit)
                        {
                            var keys = Object.keys(self.nodesToVisit);
                            var i = 0;
                            for(i =0; i!= keys.length; i+=1)
                            {
                                self._selfPatterns[keys[i]] = { children: 3 };
                            }
                        }

                        
                        self._client.updateTerritory(self._territoryId, self._selfPatterns);
                    }
                });
        };

        // This next function retrieves the relevant node information for the widget
        ParamsControl.prototype._getObjectDescriptor = function (nodeId) {
            var self = this;
            var nodeObj = self._client.getNode(nodeId);
            var obj, k;


            return new Promise(function (resolve, reject) {

                if (nodeObj) {
                    var objDescriptor = {
                        id: undefined,
                        metaName: undefined,
                        name :undefined,
                        current: false,
                        reference: false,
                        referenceInfo : {},
                        info: {},
                        use: 0,
                        last:0
                    };

                    var metaObj = self._client.getNode(nodeObj.getMetaTypeId()),
                        id = nodeObj.getId(),
                        metaName = '';

                    if (metaObj) {
                        metaName = metaObj.getAttribute(nodePropertyNames.Attributes.name);
                    }

                    if (!self._currentNodeId)
                        self.getParentFolders(nodeObj);

                    
                    if (self._currentNodeId ) {
                        
                        
                        objDescriptor.id = nodeObj.getId();
                        if ((self._currentNodeId == objDescriptor.id) && (metaName=="Params"))
                        {
                            objDescriptor.current = true;
                            objDescriptor.metaName = metaName;
                            objDescriptor.name = nodeObj.getAttribute(nodePropertyNames.Attributes.name);

                            try {
                                var definition = nodeObj.getAttribute('Definition');
                                objDescriptor.info = JSON.parse(definition);
                                objDescriptor.use = 1;
                
                            } catch (e) {
                                var estr = 'Unable to parse JSON input for parameter ' + manme  + ' in ' + hname;
                                self._logger.error(estr);
                                
                            }
                            
                            objDescriptor.use = 1;
                        }
                        else if (self.checkMetaType(metaName))
                        {
                            objDescriptor = self.updateReferenceInfo(nodeObj,metaName,objDescriptor);
                        }
                    }

                    resolve(objDescriptor);

                }
            });
        };

        ParamsControl.prototype.checkMetaName = function (metaName) {
            var self = this;
            var parentMetaTypes = ['Block','SystemModel','Environment','ExperimentSetup',
                                  'SLTrainingSetUp', 'EvaluationSetup', 'AssuranceMonitorSetup',
                                  'RLTrainingSetup','VerificationSetup','ValidationSetup','SystemIDSetup'];
            if (parentMetaTypes.indexOf(metaName) > -1)
            {
                return true;
            }
            return false;
        };

        ParamsControl.prototype.updateReferenceInfo = function (nodeObj,metaName,objDescriptor) {
            var self = this;
            var descriptor = objDescriptor;

            var childrenIds = nodeObj.getChildrenIds();
            var k =0;
            var nodeID = nodeObj.getId();
            var hname = self.getHierarchicalName(nodeID);
            
            for (k=0; k<childrenIds.length; k++)
            {
                var cnode = self._client.getNode(childrenIds[k]);
                if (!cnode)
                {
                    continue;
                }

				var cmetaObj = self._client.getNode(cnode.getMetaTypeId());
                var mname = cmetaObj.getAttribute(nodePropertyNames.Attributes.name);
                
                if (mname == 'Params')
                {
                    descriptor = self.getParamReference(cnode,descriptor,hname);

                }
                
            }

            return descriptor;
        };

        ParamsControl.prototype.getParamReference = function (nodeObj,objDescriptor,hname) {
            var self = this;
            var mname = nodeObj.getAttribute(nodePropertyNames.Attributes.name);
            var descriptor = objDescriptor;
            var definition = nodeObj.getAttribute('Definition');
            var jsonval = {};
            try {
                jsonval = JSON.parse(definition);
                var nodeid = nodeObj.getId();
                var keys = Object.keys(jsonval);
                var len = keys.length;
                if (len)
                {
                    
                    if (!(hame in descriptor.referenceInfo))
                    {
                        descriptor.referenceInfo[hame]={};
                    }
                    descriptor.referenceInfo[hame][mname]={};
                    var i = 0;
                    for (i = 0; i != len; i += 1) {
                        var pname = keys[i];
                        descriptor.referenceInfo[hame][mname][pname]=nodeid;
                    }
                    descriptor.use = 1;
                }

            } catch (e) {
                var estr = 'Unable to parse JSON input for parameter ' + manme  + ' in ' + hname;
                self._logger.error(estr);
                
            }
            return descriptor;

        };

        ParamsControl.prototype.getHierarchicalName = function (nodeid) {
            var self = this;
            var nid = nodeid;
            var nodeObj = self._client.getNode(nid);
            var pid = nodeObj.getParentId();
            var stoppingParentMetaTypes = ['BlockLibrary','BlockPackage','Systems','Assemblys','ExperimentSetup',
                                  'SLTrainingSetUp', 'EvaluationSetup', 'AssuranceMonitorSetup',
                                  'RLTrainingSetup','VerificationSetup','ValidationSetup','SystemIDSetup'];
            
            
            while (true)
            {

                if (nodeObj)
                {	
                    var cmetaObj = self._client.getNode(nodeObj.getMetaTypeId());
                    var mname = cmetaObj.getAttribute(nodePropertyNames.Attributes.name);
                    if (mname in stoppingParentMetaTypes)
                    {
                        break;
                    }
                    ret.push(nodeObj.getAttribute(nodePropertyNames.Attributes.name));
                    nid = pid;
                    nodeObj = self._client.getNode(nid);
                    if (nodeObj)
                    {
                        pid = nodeObj.getParentId();
                    }
                    else
                    {
                        break;
                    }
                }
                else{
                    break;
                }
                
            }
            if (nodeObj)
            {
                ret.push(nodeObj.getAttribute(nodePropertyNames.Attributes.name));
            }

            var ret1= ret.reverse();
            var ret2 = ret1.join('/');
            return ret2;
                
            
        };

        SelectorControl.prototype.getParentFolders = function (nodeObj) {

            var self = this;
            var ret = [];
            if (self.parentFoldersVisited)
                return ret;

            var parentNodeID, parentNode, parentMetaNode, parentMetaName;
            
            var stoppingParentMetaTypes = ['BlockLibrary','BlockPackage','Systems','Assemblys','ExperimentSetup',
                                    'SLTrainingSetUp', 'EvaluationSetup', 'AssuranceMonitorSetup',
                                    'RLTrainingSetup','VerificationSetup','ValidationSetup','SystemIDSetup'];

            var exploreMetaTypes = ['Block','System', 'Environment', 'ExperimentSetup',
                                    'SLTrainingSetUp', 'EvaluationSetup', 'AssuranceMonitorSetup',
                                    'RLTrainingSetup','VerificationSetup','ValidationSetup','SystemIDSetup'];
                
            parentNodeID = nodeObj.getParentId();
            parentNode = self._client.getNode(parentNodeID);

            var metaobj = self._client.getNode(nodeObj.getMetaTypeId());
            var metaName = metaobj.getAttribute(nodePropertyNames.Attributes.name);

            while (parentNode) {
                parentMetaNode = self._client.getNode(parentNode.getMetaTypeId());
                if (!parentMetaNode)
                    break;

                parentMetaName = parentMetaNode.getAttribute(nodePropertyNames.Attributes.name);

                if (parentMetaName in stoppingParentMetaTypes)
                {
                    break;
                }

                if (parentMetaName in exploreMetaTypes)
                {
                    self.nodesToVisit[parentNodeID]={children:3};
                }

                parentNodeID = parentNode.getParentId();
                parentNode = self._client.getNode(parentNodeID);
            }

            self.parentFoldersVisited = 1;


        };

        /* * * * * * * * Node Event Handling * * * * * * * */
        ParamsControl.prototype._eventCallback = function (events) {
            var i = events ? events.length : 0,
                event;

            this._logger.debug('_eventCallback \'' + i + '\' items');
            

            while (i--) {
                event = events[i];
                switch (event.etype) {
                    case CONSTANTS.TERRITORY_EVENT_LOAD:
                        this._logger.debug('event '+i)
                        this._onLoad(event.eid, i);
                        this.countlimit = i;
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

            this._logger.debug('event outside  '+i)

            

            this._logger.debug('_eventCallback \'' + events.length + '\' items - DONE');
        };

       

        ParamsControl.prototype._onLoad = function (gmeId, count = -1) {
            var self = this;
            this._getObjectDescriptor(gmeId)
                .then(function (description) {
                    description.last = -1;
                    if (count == 1) {
                        description.last = 1;
                    }

                    if (description.use == 1 || description.metaName || (description.last==1))
                        self._widget.addNode(description);

                });

        };

        ParamsControl.prototype._onUpdate = function (gmeId) {
            var self = this;
            self._getObjectDescriptor(gmeId)
                .then(function (description) {
                    //self._logger.debug('metaname  _onUpdate= '+ description.metaName);
                    self._widget.updateNode(description);
                });
        };

        ParamsControl.prototype._onUnload = function (gmeId) {
            this._widget.removeNode(gmeId);
        };

        ParamsControl.prototype._stateActiveObjectChanged = function (model, activeObjectId) {
            if (this._currentNodeId === activeObjectId) {
                // The same node selected as before - do not trigger
            } else {
                this.selectedObjectChanged(activeObjectId);
            }
        };

        /* * * * * * * * Visualizer life cycle callbacks * * * * * * * */
        ParamsControl.prototype.destroy = function () {
            this._detachClientEventListeners();
            this._removeToolbarItems();
        };

        ParamsControl.prototype._attachClientEventListeners = function () {
            this._detachClientEventListeners();
            WebGMEGlobal.State.on('change:' + CONSTANTS.STATE_ACTIVE_OBJECT, this._stateActiveObjectChanged, this);
        };

        ParamsControl.prototype._detachClientEventListeners = function () {
            WebGMEGlobal.State.off('change:' + CONSTANTS.STATE_ACTIVE_OBJECT, this._stateActiveObjectChanged);
        };

        ParamsControl.prototype.onActivate = function () {
            this._attachClientEventListeners();
            this._displayToolbarItems();

            if (typeof this._currentNodeId === 'string') {
                WebGMEGlobal.State.registerSuppressVisualizerFromNode(true);
                WebGMEGlobal.State.registerActiveObject(this._currentNodeId);
                WebGMEGlobal.State.registerSuppressVisualizerFromNode(false);
            }
        };

        ParamsControl.prototype.onDeactivate = function () {
            this._detachClientEventListeners();
            this._hideToolbarItems();
        };

        /* * * * * * * * * * Updating the toolbar * * * * * * * * * */
        ParamsControl.prototype._displayToolbarItems = function () {

            if (this._toolbarInitialized === true) {
                for (var i = this._toolbarItems.length; i--;) {
                    this._toolbarItems[i].show();
                }
            } else {
                this._initializeToolbar();
            }
        };

        ParamsControl.prototype._hideToolbarItems = function () {

            if (this._toolbarInitialized === true) {
                for (var i = this._toolbarItems.length; i--;) {
                    this._toolbarItems[i].hide();
                }
            }
        };

        ParamsControl.prototype._removeToolbarItems = function () {

            if (this._toolbarInitialized === true) {
                for (var i = this._toolbarItems.length; i--;) {
                    this._toolbarItems[i].destroy();
                }
            }
        };

        ParamsControl.prototype._initializeToolbar = function () {
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


    return ParamsControl;
});
