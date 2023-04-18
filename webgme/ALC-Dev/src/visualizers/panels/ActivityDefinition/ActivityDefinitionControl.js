/*globals define, WebGMEGlobal*/
/**
 * Generated by VisualizerGenerator 1.7.0 from webgme on Wed Mar 30 2022 10:22:23 GMT-0500 (CDT).
 */

define([
    'js/Constants',
    'js/Utils/GMEConcepts',
    'js/NodePropertyNames'
], function (
    CONSTANTS,
    GMEConcepts,
    nodePropertyNames
) {

    'use strict';

    function ActivityDefinitionControl(options) {

        this._logger = options.logger.fork('Control');

        this._client = options.client;

        // Initialize core collections and variables
        this._widget = options.widget;

        this._currentNodeId = null;
        this._currentNodeParentId = undefined;

        this._initWidgetEventHandlers();

        this._logger.debug('ctor finished');
    }

    ActivityDefinitionControl.prototype._initWidgetEventHandlers = function () {
        this._widget.onNodeClick = function (id) {
            // Change the current active object
            WebGMEGlobal.State.registerActiveObject(id);
        };
    };

    /* * * * * * * * Visualizer content update callbacks * * * * * * * */
    // One major concept here is with managing the territory. The territory
    // defines the parts of the project that the visualizer is interested in
    // (this allows the browser to then only load those relevant parts).
    ActivityDefinitionControl.prototype.selectedObjectChanged = function (nodeId) {
        var desc = this._getObjectDescriptor(nodeId),
            self = this;

        self._logger.debug('activeObject nodeId \'' + nodeId + '\'');

        // Remove current territory patterns
        if (self._currentNodeId) {
            self._client.removeUI(self._territoryId);
        }

        self._currentNodeId = nodeId;
        self._currentNodeParentId = undefined;

        if (typeof self._currentNodeId === 'string') {
            // Put new node's info into territory rules
            self._selfPatterns = {};
            self._selfPatterns[nodeId] = {children: 6};  // Territory "rule"

            self._widget.setTitle(desc.name.toUpperCase());

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

            var parentNodeID, parentNode, parentMetaNode, parentMetaName;
            var constructionMetaType = 'Construction';
            var nodeObj = self._client.getNode(self._currentNodeId);
            parentNodeID = nodeObj.getParentId();
            parentNode = self._client.getNode(parentNodeID);          

            while (parentNode) {
                
                parentMetaNode = self._client.getNode(parentNode.getMetaTypeId());
                if (!parentMetaNode)
                    break;

                parentMetaName = parentMetaNode.getAttribute(nodePropertyNames.Attributes.name);

                if (parentMetaName.indexOf(constructionMetaType) != -1) {
                    var constructionid = parentNodeID;
                    self._selfPatterns[constructionid] = {children: 6};
                    break;
                }

            
                parentNodeID = parentNode.getParentId();
                parentNode = self._client.getNode(parentNodeID);
            }

            // Update the territory
            self._client.updateTerritory(self._territoryId, self._selfPatterns);
        }
    };

    // This next function retrieves the relevant node information for the widget
    ActivityDefinitionControl.prototype._getObjectDescriptor = function (nodeId) {
        var node = this._client.getNode(nodeId),
            objDescriptor;
        if (node) {            
            objDescriptor = {
                id: node.getId(),
                name: node.getAttribute(nodePropertyNames.Attributes.name),
                childrenIds: node.getChildrenIds(),
                parentId: node.getParentId(),
                isConnection: GMEConcepts.isConnection(nodeId),
                ChoicesDict: this.getChoices(nodeId),
                InputsDict: this.getInputs(nodeId),
                ResultsDict: this.getResults(nodeId)
            };
        }

        return objDescriptor;
    };

    ActivityDefinitionControl.prototype.getResults = function (nodeId) {
        var parentNodeID, parentNode, parentMetaNode, parentMetaName;
        var constructionMetaType = 'Construction';
        var nodeObj = this._client.getNode(nodeId);
        parentNodeID = nodeObj.getParentId();
        parentNode = this._client.getNode(parentNodeID);
        var constructionID = null;

        
        while (parentNode) {
            parentMetaNode = this._client.getNode(parentNode.getMetaTypeId());
            if (!parentMetaNode)
                break;

            parentMetaName = parentMetaNode.getAttribute(nodePropertyNames.Attributes.name);

            if (parentMetaName.indexOf(constructionMetaType) != -1) {
                constructionID = parentNodeID;
                break;
            }
            parentNodeID = parentNode.getParentId();
            parentNode = this._client.getNode(parentNodeID);
        }

        if(constructionID == null)
            return null;

        var constructionObj = this._client.getNode(constructionID);
        var ccchildren = constructionObj.getChildrenIds();
        var expDict =  null;
        for(var k = 0;k < ccchildren.length;++k)
        {
            var cccObj = this._client.getNode(ccchildren[k]);
            if(cccObj == null)
                continue;
            //if(cccObj.getAttribute("name") === "Testing" ||
            //    cccObj.getAttribute("name") === "DataCollection")
            {
                var cccchildren = this._client.getNode(ccchildren[k]).getChildrenIds();
                for(var l = 0;l < cccchildren.length;++l)
                {
                    var ccccObj = this._client.getNode(cccchildren[l]);
                    if(ccccObj == null)
                        continue;
                    var expName = ccccObj.getAttribute("name");
                    var results = [];
                    var childMetaId = ccccObj.getMetaTypeId();
                    var cchildObj = this._client.getNode(childMetaId);
                    if(cchildObj == null)
                        continue;
                    //if(cchildObj.getAttribute("name") === "ExperimentSetup")
                    {
                        var ccccchildren = ccccObj.getChildrenIds();
                        for(var x = 0;x < ccccchildren.length;++x)
                        {
                            var ccccObj = this._client.getNode(ccccchildren[x]);
                            if(ccccObj == null)
                                continue;
                            //if(ccccObj.getAttribute("name") === "SimulationData")
                            var ccccMetaId = ccccObj.getMetaTypeId();
                            var ccccMetaObj = this._client.getNode(ccccMetaId);
                            if(ccccMetaObj == null)
                                continue;
                            var cccMetaName = ccccMetaObj.getAttribute("name");
                            if(cccMetaName.indexOf("Result") != -1)
                            {
                                var ccccchildren = ccccObj.getChildrenIds();
                                for(var x = 0;x < ccccchildren.length;++x)
                                {
                                    var ccccObj = this._client.getNode(ccccchildren[x]);
                                    if(ccccObj == null)
                                        continue;
                                    results.push([ccccObj.getId(), ccccObj.getAttribute("name")]);
                                }
                            }
                        }
                    }
                    if (expDict == null)
                    {
                        expDict = {};
                    }
                    expDict[expName] = results;
                }
            }
        }
                
        return expDict;
    };

    /*ActivityDefinitionControl.prototype.checkInclude = function (nodeChoices,choiceList) {
    {
        var include = false;
        var self = this;
        if (choiceList.length >0)
        {
            var choiceListVals = [];
            var splitstr =',';
            if (choiceList.indexOf('\n') != -1)
            {
                splitstr = '\n';
            }
            choiceListVals = choiceList.split(splitstr);
            for(var k = 0;k < choiceListVals.length;++k)
            {
                if (nodeChoices.includes(choiceListVals[k]))
                {
                    include = true;
                    break;
                }
            }
            
        }
        return include;

    };*/


    ActivityDefinitionControl.prototype.checkInclude = function (nodeChoices,choiceList) {
        
        if (choiceList.length < 1)
            return true;
        if (choiceList.length >0)
        {
            var choiceListVals = [];
            var splitstr =',';
            if (choiceList.indexOf('\n') != -1)
            {
                splitstr = '\n';
            }
            choiceListVals = choiceList.split(splitstr);
            for(var k = 0;k < choiceListVals.length;++k)
            {
                if (nodeChoices.includes(choiceListVals[k]))
                {
                    return true;
                }
            }
            
        }
        return false;
        
    };

    ActivityDefinitionControl.prototype.getInputs = function (nodeId) {
        var nodeObj = this._client.getNode(this._client.getNode(nodeId).getMetaTypeId());
        if(nodeObj == null)
        {
            return null;
        }

        if(nodeObj.getAttribute("name") === "Activity")
        {
            var childrenIds = this._client.getNode(nodeId).getChildrenIds();
            var inputDict = {};
            var inputs = [];
            var nodeChoices = this._client.getNode(nodeId).getAttribute('CurrentChoice');
            //nodeChoices = nodeChoices.replace(',','\n')
            nodeChoices = nodeChoices.split(",");
            if(nodeChoices != undefined && nodeChoices.length < 1)
                return null;

            childrenIds.map((id) => {
                var obj = this._client.getNode(id);
                if(obj == null)
                {
                    return;
                }
                var metaObj = this._client.getNode(obj.getMetaTypeId());
                if(metaObj == null)
                {
                    return;
                }

                if(metaObj.getAttribute("name") === "Input")
                {
                    var choiceList = obj.getAttribute("ChoiceList");
                    //var include = this.checkInclude(nodeChoices,choiceList);
                    var include=this.checkInclude(nodeChoices,choiceList);

                    if( include)
                    {
                        inputs.push([obj.getAttribute("name"), obj.getId(), obj.getAttribute("multi_dataset")]);
                    }
                }
            });
            inputs.sort((a, b) => a[0].localeCompare(b[0]));
            for(var i = 0;i < inputs.length;++i)
            {
                var key = inputs[i][0];
                var val = [inputs[i][1],inputs[i][2]];
                inputDict[key] = val;
            }
            return inputDict;
        }
        return null;
    };

    ActivityDefinitionControl.prototype.getChoices = function (nodeId) {
        var self = this;
        var nodeObj = this._client.getNode(this._client.getNode(nodeId).getMetaTypeId());
        if(nodeObj === null)
        {
            return null;
        }

        if(nodeObj.getAttribute("name") === "Activity")
        {
            var nodeChoices = this._client.getNode(nodeId).getAttribute('CurrentChoice');
            nodeChoices = nodeChoices.split(",");
            if(nodeChoices !== undefined && 
               nodeChoices.length < 1)
            {
                return null;
            }

            var childrenIds = this._client.getNode(nodeId).getChildrenIds();
            var choiceDict = {};
            var choiceArr = [];
            childrenIds.map((id) => 
            {
                var obj = this._client.getNode(id);
                if(obj === null)
                {
                    return;
                }

                var metaObj = self._client.getNode(obj.getMetaTypeId());
                if(metaObj.getAttribute("name") === "ParamsTable")
                {
                    var choiceList = obj.getAttribute("ChoiceList");
                    var include = this.checkInclude(nodeChoices,choiceList);
                    /*if (choiceList.length >0)
                    {
                        var choiceListVals = [];
                        var splitstr =',';
                        if (choiceList.indexOf('\n') != -1)
                        {
                            splitstr = '\n';
                        }
                        choiceListVals = choiceList.split(splitstr);
                        for(var k = 0;k < choiceListVals.length;++k)
                        {
                            if (nodeChoices.includes(choiceListVals[k]))
                            {
                                include = true;
                                break;
                            }
                        }
                        
                    }
                    if( include || choiceList.length < 1)*/
                    if (include)
                    {
                        var childrenOfChildIds = obj.getChildrenIds();
                        if(childrenOfChildIds === null)
                        {
                            return;
                        }

                        var choices = [];
                        childrenOfChildIds.map((id) => {
                            var ccObj = this._client.getNode(id);
                            if(ccObj === null)
                            {
                                return;
                            }

                            var childMetaObj = this._client.getNode(ccObj.getMetaTypeId());
                            if(childMetaObj.getAttribute("name") === "Parameter")
                            {
                                var choiceList1 = ccObj.getAttribute("ChoiceList");
                                var include1 = this.checkInclude(nodeChoices,choiceList1);
                                if (include1)
                                {
                                    var type = "value";
                                    if(ccObj.getAttribute("type") === "asset")
                                    {
                                        type = "asset";
                                    }

                                    choices.push([ccObj.getId(), 
                                        ccObj.getAttribute("name"), 
                                        ccObj.getAttribute(type),
                                        ccObj.getAttribute("value_choices").split("\n"), 
                                        ccObj.getAttribute("type"),
                                        ccObj.getAttribute("defaultValue"),
                                        ccObj.getAttribute("code_type"),
                                        ccObj.getAttribute("description"),
                                        ccObj.getAttribute("index")
                                    ]);
                                }
                            }
                        });

                        choices.sort((a, b) => b[1].localeCompare(a[1]));
                        choiceArr.push([obj.getAttribute("name"), choices]);
                    }
                }
            });
            choiceArr.sort((a, b) => a[0].localeCompare(b[0]));
            choiceArr.map((arr) => {
                choiceDict[arr[0]] = arr[1];
            });

            return choiceDict;
        }
        return null;
    };

    /* * * * * * * * Node Event Handling * * * * * * * */
    ActivityDefinitionControl.prototype._eventCallback = function (events) {
        var i = events ? events.length : 0,
            event;

        this._logger.debug('_eventCallback \'' + i + '\' items');

        while (i--) {
            event = events[i];
            switch (event.etype) {

            case CONSTANTS.TERRITORY_EVENT_LOAD:
                this._onLoad(event.eid);
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

        this._logger.debug('_eventCallback \'' + events.length + '\' items - DONE');
    };

    ActivityDefinitionControl.prototype._onLoad = function (gmeId) {
        if (this._currentNodeId === gmeId) {
            var description = this._getObjectDescriptor(gmeId);
            this._widget.addNode(description);
        }
    };

    ActivityDefinitionControl.prototype._onUpdate = function (gmeId) {
        var description = this._getObjectDescriptor(gmeId);
        this._widget.updateNode(description);
    };

    ActivityDefinitionControl.prototype._onUnload = function (gmeId) {
        this._widget.removeNode(gmeId);
    };

    ActivityDefinitionControl.prototype._stateActiveObjectChanged = function (model, activeObjectId) {
        if (this._currentNodeId === activeObjectId) {
            // The same node selected as before - do not trigger
        } else {
            this.selectedObjectChanged(activeObjectId);
        }
    };

    /* * * * * * * * Visualizer life cycle callbacks * * * * * * * */
    ActivityDefinitionControl.prototype.destroy = function () {
        this._detachClientEventListeners();
        this._removeToolbarItems();
    };

    ActivityDefinitionControl.prototype._attachClientEventListeners = function () {
        this._detachClientEventListeners();
        WebGMEGlobal.State.on('change:' + CONSTANTS.STATE_ACTIVE_OBJECT, this._stateActiveObjectChanged, this);
    };

    ActivityDefinitionControl.prototype._detachClientEventListeners = function () {
        WebGMEGlobal.State.off('change:' + CONSTANTS.STATE_ACTIVE_OBJECT, this._stateActiveObjectChanged);
    };

    ActivityDefinitionControl.prototype.onActivate = function () {
        this._attachClientEventListeners();
        this._displayToolbarItems();

        if (typeof this._currentNodeId === 'string') {
            WebGMEGlobal.State.registerActiveObject(this._currentNodeId, {suppressVisualizerFromNode: true});
        }
    };

    ActivityDefinitionControl.prototype.onDeactivate = function () {
        this._detachClientEventListeners();
        this._hideToolbarItems();
    };

    /* * * * * * * * * * Updating the toolbar * * * * * * * * * */
    ActivityDefinitionControl.prototype._displayToolbarItems = function () {

        if (this._toolbarInitialized === true) {
            for (var i = this._toolbarItems.length; i--;) {
                this._toolbarItems[i].show();
            }
        } else {
            this._initializeToolbar();
        }
    };

    ActivityDefinitionControl.prototype._hideToolbarItems = function () {

        if (this._toolbarInitialized === true) {
            for (var i = this._toolbarItems.length; i--;) {
                this._toolbarItems[i].hide();
            }
        }
    };

    ActivityDefinitionControl.prototype._removeToolbarItems = function () {

        if (this._toolbarInitialized === true) {
            for (var i = this._toolbarItems.length; i--;) {
                this._toolbarItems[i].destroy();
            }
        }
    };

    ActivityDefinitionControl.prototype._initializeToolbar = function () {
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

    return ActivityDefinitionControl;
});
