/*globals define, WebGMEGlobal*/
/**
 * Generated by VisualizerGenerator 1.7.0 from webgme on Wed Apr 11 2018 10:42:14 GMT-0500 (Central Daylight Time).
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

    var CoverageControl;

    CoverageControl = function (options) {

        this._logger = options.logger.fork('Control');

        this._client = options.client;

        // Initialize core collections and variables
        this._widget = options.widget;

        this._currentNodeId = null;
        this._currentNodeParentId = undefined;

        this._initWidgetEventHandlers();

        this._logger.debug('ctor finished');
    };

    CoverageControl.prototype._initWidgetEventHandlers = function () {
		var self = this;
        this._widget.onNodeClick = function (id) {
            // Change the current active object
            //WebGMEGlobal.State.registerActiveObject(id);
			var targetNodeObj = self._client.getNode(id);
            if (targetNodeObj) {
                self._logger.debug('*******************got object');
                if (targetNodeObj.getParentId() || targetNodeObj.getParentId() === CONSTANTS.PROJECT_ROOT_ID) {
                    self._logger.debug('*******************got parent id');
                    WebGMEGlobal.State.registerActiveObject(targetNodeObj.getParentId());
                    WebGMEGlobal.State.registerActiveSelection([id]);
                    WebGMEGlobal.State.registerActiveVisualizer('ModelEditor');
                } 
	     
            }
			else {
				self._logger.debug('*******************no object');
			}
        };
		
		this._widget.onEditGSNCoverage = function (id, newValues) {
            self._logger.debug(' on edit onEditGSNCoverage');
			var newgsncoverage = JSON.stringify(newValues);
            self._client.setAttributes(id, 'GSNCoverage', newgsncoverage);
	   
        };
    };

    /* * * * * * * * Visualizer content update callbacks * * * * * * * */
    // One major concept here is with managing the territory. The territory
    // defines the parts of the project that the visualizer is interested in
    // (this allows the browser to then only load those relevant parts).
    CoverageControl.prototype.selectedObjectChanged = function (nodeId) {
         var desc =  undefined,
                self = this;
        self._getObjectDescriptor(nodeId)
            .then(function(desc) {

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
				self._selfPatterns[nodeId] = {children: 0};  // Territory "rule"

				//self._widget.setTitle("Coverage Results");

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

				self._selfPatterns[nodeId] = {children: 1};
				self._client.updateTerritory(self._territoryId, self._selfPatterns);
			}
        });
    };

    // This next function retrieves the relevant node information for the widget
    CoverageControl.prototype._getObjectDescriptor = function (nodeId) {
        var  self = this,
			node = this._client.getNode(nodeId),
            objDescriptor;
			
		return new Promise(function(resolve,reject) {
			
			if (node) {
				objDescriptor = {
					id: node.getId(),
					name: node.getAttribute(nodePropertyNames.Attributes.name),
					use: 0,
					metaname:     '',
					comptype:     '',
					compinfo:     '',
					fninfo:		  '',
					compcoverage: '',
					fncoverage:   '',
					gsncoverage:  ''
				};
				var metaObj = self._client.getNode(node.getMetaTypeId());
                if (metaObj) {
                    objDescriptor.metaname = metaObj.getAttribute(nodePropertyNames.Attributes.name);
                
					if (objDescriptor.metaname.indexOf('Folder') != -1)
					{
						objDescriptor.comptype = JSON.parse(node.getAttribute('CompTypes'));
						var val = node.getAttribute('CompTypes')
						self._logger.debug('Comp Types: '+val);
						self._logger.debug('Comp Types afer parse: '+objDescriptor.comptype);
						
						objDescriptor.compinfo = JSON.parse(node.getAttribute('CompInfo'));
						objDescriptor.fninfo = JSON.parse(node.getAttribute('FnInfo'));
						objDescriptor.compcoverage = JSON.parse(node.getAttribute('CompCoverage'));
						objDescriptor.fncoverage = JSON.parse(node.getAttribute('FnCoverage'));
						objDescriptor.gsncoverage = JSON.parse(node.getAttribute('GSNCoverage'));
						objDescriptor.use = 1;
					}
				}
				resolve(objDescriptor);
				
			}
		});

      
    };
	
	
    /* * * * * * * * Node Event Handling * * * * * * * */
    CoverageControl.prototype._eventCallback = function (events) {
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

    CoverageControl.prototype._onLoad = function (gmeId) {
		var self = this;
		if (gmeId == self._currentNodeId)
		{
			self._getObjectDescriptor(gmeId)
                .then(function(description) {
					if (description && description.use ==1)
					{
						self._widget.addNode(description);
						//self._widget.setEditable();
					}
                });
		}
    };

    CoverageControl.prototype._onUpdate = function (gmeId) {
		var self = this;
		if (gmeId == self._currentNodeId)
		{
			self._getObjectDescriptor(gmeId)
                .then(function(description) {
                    self._widget.updateNode(description);
                });
		}
    };

    CoverageControl.prototype._onUnload = function (gmeId) {
		if (gmeId == self._currentNodeId)
		{
			this._widget.removeNode(gmeId);
		}
    };

    CoverageControl.prototype._stateActiveObjectChanged = function (model, activeObjectId) {
        if (this._currentNodeId === activeObjectId) {
            // The same node selected as before - do not trigger
        } else {
            this.selectedObjectChanged(activeObjectId);
        }
    };

    /* * * * * * * * Visualizer life cycle callbacks * * * * * * * */
    CoverageControl.prototype.destroy = function () {
        this._detachClientEventListeners();
        this._removeToolbarItems();
    };

    CoverageControl.prototype._attachClientEventListeners = function () {
        this._detachClientEventListeners();
        WebGMEGlobal.State.on('change:' + CONSTANTS.STATE_ACTIVE_OBJECT, this._stateActiveObjectChanged, this);
    };

    CoverageControl.prototype._detachClientEventListeners = function () {
        WebGMEGlobal.State.off('change:' + CONSTANTS.STATE_ACTIVE_OBJECT, this._stateActiveObjectChanged);
    };

    CoverageControl.prototype.onActivate = function () {
        this._attachClientEventListeners();
        this._displayToolbarItems();

        if (typeof this._currentNodeId === 'string') {
            WebGMEGlobal.State.registerActiveObject(this._currentNodeId, {suppressVisualizerFromNode: true});
        }
    };

    CoverageControl.prototype.onDeactivate = function () {
        this._detachClientEventListeners();
        this._hideToolbarItems();
    };

    /* * * * * * * * * * Updating the toolbar * * * * * * * * * */
    CoverageControl.prototype._displayToolbarItems = function () {

        if (this._toolbarInitialized === true) {
            for (var i = this._toolbarItems.length; i--;) {
                this._toolbarItems[i].show();
            }
        } else {
            this._initializeToolbar();
        }
    };

    CoverageControl.prototype._hideToolbarItems = function () {

        if (this._toolbarInitialized === true) {
            for (var i = this._toolbarItems.length; i--;) {
                this._toolbarItems[i].hide();
            }
        }
    };

    CoverageControl.prototype._removeToolbarItems = function () {

        if (this._toolbarInitialized === true) {
            for (var i = this._toolbarItems.length; i--;) {
                this._toolbarItems[i].destroy();
            }
        }
    };

    CoverageControl.prototype._initializeToolbar = function () {
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

    return CoverageControl;
});